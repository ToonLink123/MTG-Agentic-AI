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
from .rules import INTERACTION_SCHEMA, RulesStore
from .schema import SchemaError
from .scryfall import ScryfallClient

log = logging.getLogger(__name__)


class DeckReviewer:
    def __init__(
        self,
        brackets: BracketChecker,
        rules: RulesStore,
        model: JSONModel,
        store: OracleStore,
        client: ScryfallClient,
    ):
        self.brackets, self.rules, self.model = brackets, rules, model
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
            jobs.append(
                (
                    combo["cards"],
                    (
                        "Does this Commander Spellbook interaction work under these exact prerequisites? "
                        f"Prerequisites: {combo['prerequisites']}. States: {combo['uses']}. "
                        f"Steps: {combo['steps']}. Claimed results: {combo['results']}. "
                        "Do not infer the starting state can be reached or assign a win turn."
                    ),
                )
            )
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
            requests.append(self.rules.interaction_request(names, question))
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
        errors = []
        if bracket_report["verdict"] != "fits":
            errors.append(
                {
                    "error": "bracket_review",
                    "detail": bracket_report["verdict"],
                    "reasons": bracket_report["reasons"],
                    "instruction": "Remove flagged cards/combos or revise within the fixed target bracket.",
                }
            )
        for result in interactions:
            if result["status"] != "supported":
                errors.append(
                    {
                        "error": "interaction_review",
                        "cards": result["cards"],
                        "detail": result["status"],
                        "question": result["question"],
                        "claims": result["claims"],
                        "instruction": "Do not rely on this unsupported/contradicted interaction. "
                        "Choose an evidenced plan; never fabricate citations.",
                    }
                )
        return {
            "status": "passed" if not errors else "needs_revision",
            "bracket": bracket_report,
            "rulings_ingestion": ingestion,
            "interactions": interactions,
            "errors": errors,
        }

    def _ask(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        try:
            return request_json(self.model, messages, INTERACTION_SCHEMA)
        except SchemaError:
            return None  # A malformed answer fails its own question, not the whole review.
