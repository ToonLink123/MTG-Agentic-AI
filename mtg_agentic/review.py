"""Safety review shared by drafts, standalone checks, and evaluations."""

from __future__ import annotations

from typing import Any

from .brackets import BracketChecker
from .cards import name_key
from .llm import JSONModel
from .oracle import OracleStore
from .rules import RulesStore
from .scryfall import ScryfallClient


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

    def check(self, cards: dict[str, int], commander: str, bracket: int) -> dict[str, Any]:
        bracket_report = self.brackets.check(cards, commander, bracket)
        all_names = list(dict.fromkeys([commander, *cards]))
        ingestion = self.rules.ingest_deck(all_names, self.store, self.client)
        records = [self.store.get_card(n) for n in all_names]
        proliferators = [
            c["name"] for c in records if c and "proliferate" in c["oracle_text"].lower()
        ]
        myojin = [c["name"] for c in records if c and c["name"].startswith("Myojin ")]
        jobs: list[tuple[list[str], str]] = []
        jodah = name_key(commander) == name_key("Jodah, Archmage Eternal")
        for name in myojin:
            involved = ([commander] if jodah else []) + [name] + proliferators[:1]
            jobs.append(
                (
                    involved,
                    (
                        f"For {name}, identify exactly when it enters with a divinity counter. "
                        + (
                            "Does paying Jodah's alternate cost while casting from hand preserve that "
                            "condition, and does casting from another zone or putting it onto the "
                            "battlefield satisfy it? "
                            if jodah
                            else ""
                        )
                        + (
                            "Can proliferate add a divinity counter if none is already present, "
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
        interactions = []
        for i, (names, question) in enumerate(jobs):
            if i >= 20:
                interactions.append(
                    {
                        "status": "unsupported",
                        "cards": names,
                        "question": question,
                        "claims": [],
                        "errors": ["Review budget exceeded; manual review required."],
                    }
                )
            else:
                interactions.append(self.rules.check_interaction(names, question, self.model))
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
