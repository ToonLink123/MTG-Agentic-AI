"""Safety review shared by drafts, standalone checks, and evaluations."""

from __future__ import annotations

import logging
import threading
from functools import partial
from typing import Any

from . import activity
from .brackets import BracketChecker
from .cards import name_key
from .llm import JSONModel, concurrency, parallel, request_json
from .oracle import OracleStore
from .rules import EVIDENCE, INTERACTION_SCHEMA, REVIEW_PASSES, RulesStore
from .schema import SchemaError
from .scryfall import ScryfallClient

log = logging.getLogger(__name__)
# Commander Spellbook's zone letters for where a combo piece starts.
ZONES = {
    "B": "on the battlefield", "H": "in your hand", "G": "in your graveyard", "E": "in exile",
    "L": "in your library", "C": "in the command zone",
}


def combo_question(combo: dict[str, Any]) -> str:
    """A Commander Spellbook combo as a plain rules question. The raw zone and state fields
    would only add noise words to the rules lookup and the judge's reading. The steps come
    first: the lookup also ranks rules for each of the first six sentences."""
    start = []
    for use in combo["uses"]:
        zones = " or ".join(ZONES.get(z, z) for z in use.get("zoneLocations") or [])
        states = [v.strip() for k, v in use.items() if k.endswith("CardState") and v and v.strip()]
        start.append(" ".join(filter(None, [
            use["card"]["name"], zones, f"({'; '.join(states)})" if states else "",
        ])))
    prerequisites = " ".join(
        " ".join(p.split()) for p in combo["prerequisites"] if isinstance(p, str) and p.strip()
    )
    return "\n".join(filter(None, [
        f"Steps: {' '.join(combo['steps'].split())}",
        f"Prerequisites: {prerequisites}" if prerequisites else "",
        f"Starting state: {'; '.join(start)}." if start else "",
        f"Claimed results: {'; '.join(combo['results'])}.",
        # Rules lookup reads words as rules terms: "reached" finds the Reach keyword, and
        # "Commander Spellbook" the commander rules, so the wording avoids both.
        "Does this combo work as described? Check each step against the rules: each "
        "ability's costs and timing, what each triggered ability does, and whether the steps "
        "can repeat. Do not assume the starting state is achievable or assign a win turn.",
    ]))



class DeckReviewer:
    def __init__(
        self,
        brackets: BracketChecker,
        rules: RulesStore,
        model: JSONModel,
        store: OracleStore,
        client: ScryfallClient,
        *,
        evidence: int = EVIDENCE,
        passes: int = REVIEW_PASSES,
    ):
        self.brackets, self.rules, self.model = brackets, rules, model
        self.evidence = evidence  # Evidence items per rules question.
        self.passes = passes  # Answers per question, while the judge names missing rules.
        self.store, self.client = store, client

    @property
    def game_changer_names(self) -> list[str]:
        return self.brackets.game_changer_list["names"]

    def check(
        self, cards: dict[str, int], commander: str, bracket: int,
        *, bracket3_mode: str = "standard",
    ) -> dict[str, Any]:
        log.info("review: bracket check")
        bracket_report = self.brackets.check(cards, commander, bracket, bracket3_mode=bracket3_mode)
        all_names = list(dict.fromkeys([commander, *cards]))
        log.info("review: rulings for %d cards", len(all_names))
        ingestion = self.rules.ingest_deck(all_names, self.store, self.client)
        records = [self.store.get_card(n) for n in all_names]
        proliferators = [
            c["name"] for c in records if c and "proliferate" in c["oracle_text"].lower()
        ]
        myojin = [c["name"] for c in records if c and c["name"].startswith("Myojin ")]
        jobs: list[tuple[list[str], str]] = []
        jodah = name_key(commander) == name_key("Jodah, Archmage Eternal")
        for name in myojin:
            card = next(c for c in records if c and c["name"] == name)
            counter = "indestructible" if "indestructible counter" in card["oracle_text"].lower() else "divinity"
            involved = ([commander] if jodah else []) + [name] + proliferators[:1]
            jobs.append(
                (
                    involved,
                    (
                        f"For {name}, identify exactly when it enters with a {counter} counter. "
                        + (
                            "Does paying Jodah's alternate cost while casting from hand preserve that "
                            "condition, and does casting from another zone or putting it onto the "
                            "battlefield satisfy it? "
                            if jodah
                            else ""
                        )
                        + (
                            f"Can proliferate add a {counter} counter if none is already present, "
                            "and can it add another while one remains? "
                            if proliferators
                            else ""
                        )
                        + "Account for costs, replacement effects, and the exact Oracle wording."
                    ),
                )
            )
        for combo in bracket_report["spellbook"]["combos"]:
            jobs.append((combo["cards"], combo_question(combo)))
        if proliferators and not myojin:
            jobs.append(
                (
                    proliferators[:1],
                    (
                        "Can proliferate give a player their first poison counter, or increase poison "
                        "counters that player already has? Cite the current definition."
                    ),
                )
            )
        # Retrieval stays on this thread (SQLite, embedder); every question is then asked at
        # once, up to the model's concurrency, and each answer is checked in order.
        requests = []
        for i, (names, question) in enumerate(jobs, 1):
            short = " + ".join(n.split(",")[0] for n in names)
            log.info("review: rules question %d/%d (%s)", i, len(jobs), short)
            requests.append(self.rules.interaction_request(names, question, limit=self.evidence))
        workers = concurrency(self.model)
        answers: list[dict[str, Any] | None] = []
        if jobs:
            log.info("review: asking %d rules question(s), up to %d at a time", len(jobs), workers)
            lock, answered = threading.Lock(), [0]

            def ask(messages: list[dict[str, str]]) -> dict[str, Any] | None:
                answer = self._ask(messages)
                with lock:
                    answered[0] += 1
                    count = answered[0]
                activity.progress("judges", f"{count}/{len(jobs)} answered")
                return answer

            task = f"{len(jobs)} rules questions, up to {workers} at a time"
            with activity.agent("judges", self.model, task):
                answers = parallel([partial(ask, messages) for messages, _ in requests], workers)
        # Another look for each answer left unsupported by a rule it named as missing, round by
        # round: the lookups run here, then those questions are asked again at once.
        looks: list[list[dict[str, Any]]] = [[] for _ in jobs]
        stopped: set[int] = set()
        for round_number in range(self.passes - 1):
            final = round_number == self.passes - 2
            retries = {}
            for i, ((names, question), (_, evidence), answer) in enumerate(zip(jobs, requests, answers)):
                if answer is not None and i not in stopped:
                    nxt = self.rules.next_look(
                        names, question, evidence, answer, looks[i], limit=self.evidence, final=final,
                    )
                    if nxt is not None:
                        retries[i] = nxt
            if not retries:
                break
            log.info("review: another look for %d unverified question(s)", len(retries))
            task = f"another look at {len(retries)} unverified question(s)"
            with activity.agent("judges", self.model, task):
                finals = parallel(
                    [partial(self._ask, messages) for messages, _, _ in retries.values()],
                    workers,
                )
            for (i, (messages, evidence, look)), again in zip(retries.items(), finals):
                if again is None:
                    stopped.add(i)  # A malformed answer keeps the last good one, with its evidence.
                    continue
                looks[i].append(look)
                requests[i], answers[i] = (messages, evidence), {**again, "looks": looks[i]}
        interactions = []
        for (names, question), (_, evidence), answer in zip(jobs, requests, answers):
            if answer is None:
                interactions.append({
                    "status": "unsupported", "claims": [], "cards": names, "question": question,
                    "evidence": evidence,
                    "errors": ["The model returned invalid grounded JSON for this question."],
                })
            else:
                interactions.append(
                    self.rules.judge_interaction(names, question, evidence, answer)
                )
        errors, warnings = [], []
        verdict, flags = bracket_report["verdict"], bracket_report.get("flags", [])
        if verdict == "mismatch":
            errors.append(
                {
                    "error": "bracket_review",
                    "detail": verdict,
                    "reasons": bracket_report["reasons"],
                    "instruction": "Remove flagged cards/combos or revise within the fixed target bracket.",
                }
            )
        elif flags:
            # Uncertain, but about named cards the drafter can cut: still a problem to fix.
            errors.append(
                {
                    "error": "bracket_review",
                    "detail": verdict,
                    "reasons": [flag["reason"] for flag in flags],
                    "cards": sorted({name for flag in flags for name in flag["cards"]}),
                    "instruction": "Remove a card from each flagged combo or revise within the fixed target bracket.",
                }
            )
        elif verdict == "uncertain":
            # Nothing it names is a card to cut (a win-speed estimate, a truncated combo
            # search), so no repair could fix it: the player sees it instead.
            warnings.append({
                "warning": "bracket_uncertain",
                "detail": "The bracket check is uncertain, but names no card to change.",
                "reasons": (bracket_report.get("warnings") or bracket_report["reasons"])[:6],
            })
        for result in interactions:
            if result["status"] == "contradicted":
                errors.append(
                    {
                        "error": "interaction_review",
                        "cards": result["cards"],
                        "detail": result["status"],
                        "question": result["question"],
                        "claims": result["claims"],
                        "instruction": "Do not rely on this contradicted interaction. "
                        "Choose an evidenced plan; never fabricate citations.",
                    }
                )
            elif result["status"] == "unsupported":
                # The judge could not prove it from the retrieved rules; that is not evidence
                # the interaction fails, so it is reported but does not block the deck.
                warnings.append({
                    "warning": "interaction_unverified",
                    "cards": result["cards"],
                    "question": result["question"],
                    "detail": "The rules judge could not verify this from the retrieved rules "
                              "and rulings; it was not shown to fail.",
                    **({"missing": result["missing"]} if result.get("missing") else {}),
                })
        return {
            "status": "passed" if not errors else "needs_revision",
            "bracket": bracket_report,
            "rulings_ingestion": ingestion,
            "interactions": interactions,
            "errors": errors,
            "warnings": warnings,
        }

    def _ask(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        try:
            return request_json(self.model, messages, INTERACTION_SCHEMA)
        except SchemaError:
            return None  # A malformed answer fails its own question, not the whole review.
