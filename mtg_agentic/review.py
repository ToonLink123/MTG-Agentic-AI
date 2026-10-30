"""Safety review shared by drafts, standalone checks, evaluations and engine maps.

The rules questions a review asks come from the deck itself: every Commander Spellbook combo in
it, and the engine map's own links between the deck's cards (what the drafters built the deck
around), not questions written for one commander. Judge answers them with cited evidence. A
judge with native tools (Claude) looks up the rules a step still needs by number, by topic or in
the cards' rulings before it answers; any other model gets one more retrieval for a rule it
names as missing. Settled answers go into the rules store's ledger, keyed by the cards' text,
the question and the rules version, so a repair attempt, an edit or a later session never asks
the same settled question again.
"""

from __future__ import annotations

import logging
import threading
from functools import partial
from typing import Any

from . import activity
from .agent import supports_tools
from .brackets import BracketChecker
from .cards import name_key
from .llm import JSONModel, concurrency, parallel, request_json
from .oracle import OracleStore
from .rules import EVIDENCE, INTERACTION_SCHEMA, REVIEW_PASSES, RulesStore
from .schema import SchemaError
from .scryfall import ScryfallClient

# Engine-map links a review or a map check asks about, most central first.
LINK_QUESTIONS = 6

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



def link_question(link: dict[str, Any]) -> str:
    """An engine-map link as a rules question about its two cards."""
    how = " ".join(link["how"].split()).rstrip(".")
    return (
        f"{link['from']} and {link['to']}: {how}. Does this interaction work as described? "
        "Check each step against the rules: each ability's costs and timing, what each "
        "triggered or static ability does, and any condition it needs."
    )


def link_jobs(
    engine_map: dict[str, Any] | None, names: set[str], *, limit: int = LINK_QUESTIONS,
) -> list[tuple[list[str], str, dict[str, Any]]]:
    """The map's own links whose cards are all in `names` (by name key), as (cards, question,
    link). Combo links are left to the Spellbook combo questions; links touching the map's
    primary cards come first."""
    if not engine_map:
        return []
    primary = {
        name_key(card["name"]) for stage in engine_map.get("stages", [])
        for card in stage["cards"] if card["tier"] == "primary"
    }
    links = [
        link for link in engine_map.get("links", [])
        if link["source"] != "combo"
        and {name_key(link["from"]), name_key(link["to"])} <= names
    ]
    links.sort(key=lambda link: -len({name_key(link["from"]), name_key(link["to"])} & primary))
    return [([link["from"], link["to"]], link_question(link), link) for link in links[:limit]]


class Judge:
    """Rules questions about cards, answered from cited evidence and kept in the ledger."""

    def __init__(
        self, rules: RulesStore, model: JSONModel, store: OracleStore, client: ScryfallClient,
        *, evidence: int = EVIDENCE, passes: int = REVIEW_PASSES,
    ):
        self.rules, self.model, self.store, self.client = rules, model, store, client
        self.evidence = evidence  # Evidence items per rules question.
        self.passes = passes  # Answers per question, while the judge names missing rules.

    def key(self, cards: list[str], question: str) -> str:
        texts = [(name, (self.store.get_card(name) or {}).get("oracle_text", "")) for name in cards]
        return self.rules.ledger_key(texts, question)

    def answer(self, jobs: list[tuple[list[str], str]], *, ingest: bool = True) -> list[dict[str, Any]]:
        """One checked answer per (cards, question), in order. Rulings for every card are
        synced first (unless the caller already did); settled answers come from the ledger
        without a model call."""
        names = list(dict.fromkeys(name for cards, _ in jobs for name in cards))
        if names and ingest:
            log.info("review: rulings for %d cards", len(names))
            self.rules.ingest_deck(names, self.store, self.client)
        keys = [self.key(cards, question) for cards, question in jobs]
        results: list[dict[str, Any] | None] = [self.rules.recall(key) for key in keys]
        if any(results):
            log.info("review: %d of %d rules question(s) already settled in the ledger",
                     sum(1 for r in results if r), len(jobs))
        todo = [i for i, result in enumerate(results) if result is None]
        if todo:
            answered = self._with_tools(jobs, todo) if supports_tools(self.model) else self._asked(jobs, todo)
            for i, result in zip(todo, answered):
                results[i] = result
                self.rules.remember(keys[i], result)
        return [result for result in results if result is not None]

    def _with_tools(self, jobs: list[tuple[list[str], str]], todo: list[int]) -> list[dict[str, Any]]:
        workers = concurrency(self.model)
        log.info("review: %d rules question(s) for judges with tools, up to %d at a time", len(todo), workers)

        def ask(cards: list[str], question: str) -> dict[str, Any]:
            try:
                return self.rules.judge_with_tools(
                    cards, question, self.model, limit=self.evidence, max_steps=2 + 2 * self.passes,
                )
            except SchemaError:
                return _failed(cards, question, [])

        return parallel([partial(ask, *jobs[i]) for i in todo], workers)

    def _asked(self, jobs: list[tuple[list[str], str]], todo: list[int]) -> list[dict[str, Any]]:
        # Retrieval stays on this thread (SQLite, embedder); every question is then asked at
        # once, up to the model's concurrency, and each answer is checked in order.
        requests = []
        for number, i in enumerate(todo, 1):
            names, question = jobs[i]
            short = " + ".join(n.split(",")[0] for n in names)
            log.info("review: rules question %d/%d (%s)", number, len(todo), short)
            requests.append(self.rules.interaction_request(names, question, limit=self.evidence))
        workers = concurrency(self.model)
        log.info("review: asking %d rules question(s), up to %d at a time", len(todo), workers)
        lock, answered = threading.Lock(), [0]

        def ask(messages: list[dict[str, str]]) -> dict[str, Any] | None:
            answer = self._ask(messages)
            with lock:
                answered[0] += 1
                count = answered[0]
            activity.progress("judges", f"{count}/{len(todo)} answered")
            return answer

        task = f"{len(todo)} rules questions, up to {workers} at a time"
        with activity.agent("judges", self.model, task):
            answers = parallel([partial(ask, messages) for messages, _ in requests], workers)
        # Another look for each answer left unsupported by a rule it named as missing, round by
        # round: the lookups run here, then those questions are asked again at once.
        looks: list[list[dict[str, Any]]] = [[] for _ in todo]
        stopped: set[int] = set()
        for round_number in range(self.passes - 1):
            final = round_number == self.passes - 2
            retries = {}
            for n, (i, (_, evidence), answer) in enumerate(zip(todo, requests, answers)):
                if answer is not None and n not in stopped:
                    names, question = jobs[i]
                    nxt = self.rules.next_look(
                        names, question, evidence, answer, looks[n], limit=self.evidence, final=final,
                    )
                    if nxt is not None:
                        retries[n] = nxt
            if not retries:
                break
            log.info("review: another look for %d unverified question(s)", len(retries))
            task = f"another look at {len(retries)} unverified question(s)"
            with activity.agent("judges", self.model, task):
                finals = parallel(
                    [partial(self._ask, messages) for messages, _, _ in retries.values()], workers,
                )
            for (n, (messages, evidence, look)), again in zip(retries.items(), finals):
                if again is None:
                    stopped.add(n)  # A malformed answer keeps the last good one, with its evidence.
                    continue
                looks[n].append(look)
                requests[n], answers[n] = (messages, evidence), {**again, "looks": looks[n]}
        results = []
        for i, (_, evidence), answer in zip(todo, requests, answers):
            names, question = jobs[i]
            if answer is None:
                results.append(_failed(names, question, evidence))
            else:
                results.append(self.rules.judge_interaction(names, question, evidence, answer))
        return results

    def _ask(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        try:
            return request_json(self.model, messages, INTERACTION_SCHEMA)
        except SchemaError:
            return None  # A malformed answer fails its own question, not the whole review.


def _failed(cards: list[str], question: str, evidence: list[Any]) -> dict[str, Any]:
    return {
        "status": "unsupported", "claims": [], "cards": cards, "question": question,
        "evidence": evidence, "errors": ["The model returned invalid grounded JSON for this question."],
    }


def check_links(engine_map: dict[str, Any], judge: Judge, commander: str) -> dict[str, Any]:
    """The engine map with its central links checked against the rules before anyone drafts:
    a link the rules contradict leaves the map and becomes a gap, so no drafter builds on it;
    the others are marked supported or unverified."""
    names = {name_key(commander)} | {
        name_key(card["name"]) for stage in engine_map.get("stages", []) for card in stage["cards"]
    }
    names |= {name_key(n) for link in engine_map.get("links", []) for n in (link["from"], link["to"])}
    jobs = link_jobs(engine_map, names)
    if not jobs:
        return engine_map
    results = judge.answer([(cards, question) for cards, question, _ in jobs])
    verdicts = {id(link): result["status"] for (_, _, link), result in zip(jobs, results)}
    links, gaps = [], list(engine_map.get("gaps", []))
    for link in engine_map.get("links", []):
        status = verdicts.get(id(link))
        if status == "contradicted":
            gaps.insert(0, f"The rules contradict {link['from']} -> {link['to']}: {link['how'][:120]}")
            continue
        verified = (
            link.get("verified", "unchecked") if status is None
            else "supported" if status == "supported" else "unverified"
        )
        links.append({**link, "verified": verified})
    log.info("engine map: %d link(s) checked against the rules, %d contradicted", len(jobs),
             sum(1 for s in verdicts.values() if s == "contradicted"))
    return {**engine_map, "links": links, "gaps": gaps[:8]}


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
        self.store, self.client = store, client
        self.judge = Judge(rules, model, store, client, evidence=evidence, passes=passes)

    @property
    def evidence(self) -> int:
        return self.judge.evidence

    @property
    def passes(self) -> int:
        return self.judge.passes

    @property
    def game_changer_names(self) -> list[str]:
        return self.brackets.game_changer_list["names"]

    def check(
        self, cards: dict[str, int], commander: str, bracket: int,
        *, bracket3_mode: str = "standard", engine_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        log.info("review: bracket check")
        bracket_report = self.brackets.check(cards, commander, bracket, bracket3_mode=bracket3_mode)
        all_names = list(dict.fromkeys([commander, *cards]))
        jobs: list[tuple[list[str], str]] = [
            (combo["cards"], combo_question(combo)) for combo in bracket_report["spellbook"]["combos"]
        ]
        # The engine the deck was built around, wherever both cards of a map link made it in.
        jobs += [(names, question) for names, question, _ in
                 link_jobs(engine_map, {name_key(n) for n in all_names})]
        # Every card's rulings are synced, even with nothing to ask, as the review has always done.
        log.info("review: rulings for %d cards", len(all_names))
        ingestion = self.rules.ingest_deck(all_names, self.store, self.client)
        interactions = self.judge.answer(jobs, ingest=False) if jobs else []
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
