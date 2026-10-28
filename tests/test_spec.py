import copy
import json
import tempfile
import unittest
from pathlib import Path

from mtg_agentic.oracle import OracleStore
from mtg_agentic.schema import SPEC_SCHEMA, SchemaError, load_json, validate_json
from mtg_agentic.spec import DeckSpec, ground_spec, intake
from tests.fixtures import card
from tests.pipeline_fixtures import JODAH, FakeCards, FakeModel, make_spec, records


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = OracleStore(Path(self.directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = records()
        self.store.import_cards(self.records)

    def test_schema_matches_jodah_example_and_roundtrips(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "jodah-spec.json"
        data = load_json(example.read_text(encoding="utf-8"))
        validate_json(data, SPEC_SCHEMA)
        self.assertEqual(DeckSpec.from_dict(data).to_dict(), make_spec().to_dict())
        self.assertEqual(ground_spec(make_spec(), self.store), make_spec())

    def test_old_specs_default_to_standard_and_custom_modes_roundtrip(self):
        data = make_spec().to_dict()
        data.pop("bracket3_mode")
        self.assertEqual(DeckSpec.from_dict(data).bracket3_mode, "standard")
        for mode in ("unrestricted", "limited-land-destruction"):
            spec = make_spec(bracket3_mode=mode)
            self.assertEqual(DeckSpec.from_dict(spec.to_dict()), spec)
            with self.assertRaises(SchemaError):
                make_spec(bracket=2, bracket3_mode=mode)
        with self.assertRaises(SchemaError):
            make_spec(bracket3_mode="unknown")

    def test_explicit_intake_mode_overrides_model_and_is_in_payload(self):
        model = FakeModel(make_spec().to_dict())
        spec = intake(
            "Myojin", JODAH["name"], model, FakeCards(self.records), self.store,
            bracket3_mode="unrestricted",
        )
        self.assertEqual(spec.bracket3_mode, "unrestricted")
        payload = json.loads(model.calls[0][0][1]["content"])
        self.assertEqual(payload["bracket3_mode"], "unrestricted")

    def test_missing_extra_and_wrong_types_are_rejected(self):
        original = make_spec().to_dict()
        mutations = []
        missing = dict(original)
        missing.pop("themes")
        mutations.append(missing)
        mutations.extend(
            [
                {**original, "unexpected": True},
                {**original, "bracket": True},
                {**original, "color_identity": "GRU"},
                {**original, "themes": "tribal"},
                {**original, "themes": [1]},
                {**original, "playstyle": "   "},
                {**original, "role_targets": {**original["role_targets"], "lands": 100}},
                {**original, "role_targets": {**original["role_targets"], "ramp": 10.0}},
            ]
        )
        for data in mutations:
            with self.subTest(data=data), self.assertRaises(SchemaError):
                DeckSpec.from_dict(data)

    def test_conflicts_duplicates_and_commander_constraints(self):
        for changes in (
            {"must_include": ["Spell 1", "spell 1"]},
            {"must_include": ["Spell 1"], "exclude": ["SPELL 1"]},
            {"must_include": [JODAH["name"]]},
            {"exclude": [JODAH["name"]]},
        ):
            with self.subTest(changes=changes), self.assertRaises(SchemaError):
                make_spec(**changes)

    def test_grounding_rejects_identity_unknown_cards_and_illegal_includes(self):
        bad = card("Illegal", legal="banned")
        self.store.import_cards([*self.records, bad])
        for spec in (
            make_spec(color_identity="WU"),
            make_spec(commander="Invented Commander"),
            make_spec(must_include=["Made Up"]),
            make_spec(exclude=["Made Up"]),
            make_spec(must_include=["Illegal"]),
        ):
            with self.subTest(spec=spec), self.assertRaises(SchemaError):
                ground_spec(spec, self.store)

    def test_intake_uses_fetched_oracle_text_and_strict_schema(self):
        model = FakeModel(make_spec().to_dict())
        client = FakeCards(self.records)
        spec = intake(
            "Myojin tribal with divinity counters and proliferate",
            JODAH["name"],
            model,
            client,
            self.store,
        )
        self.assertEqual(spec, make_spec())
        self.assertEqual(client.get_calls, [(JODAH["name"], False)])
        payload = json.loads(model.calls[0][0][1]["content"])
        self.assertEqual(payload["commander_card"]["oracle_text"], JODAH["oracle_text"])
        self.assertNotIn("prices", payload["commander_card"])

    def test_intake_rejects_changed_commander_or_hallucinated_identity(self):
        other = card("Other Commander", types="Legendary Creature — Human", colors="WUBRG")
        self.store.import_cards([*self.records, other])
        for result in (
            make_spec(commander="Other Commander").to_dict(),
            make_spec(color_identity="WU").to_dict(),
            {},
        ):
            with self.subTest(result=result), self.assertRaises(SchemaError):
                intake(
                    "Myojin", JODAH["name"], FakeModel(result), FakeCards(self.records), self.store
                )

    def test_intake_requires_request_and_existing_snapshot(self):
        for request in ("", " "):
            with self.assertRaises(ValueError):
                intake(request, JODAH["name"], FakeModel(), FakeCards(self.records), self.store)
        with OracleStore(Path(self.directory.name) / "empty.sqlite3") as empty:
            with self.assertRaises(ValueError):
                intake("Myojin", JODAH["name"], FakeModel(), FakeCards(self.records), empty)

    def test_schema_does_not_mutate_caller_data(self):
        data = make_spec().to_dict()
        old = copy.deepcopy(data)
        result = DeckSpec.from_dict(data)
        result.themes.append("new theme")
        self.assertEqual(data, old)

    def test_json_rejects_duplicate_keys_nonfinite_prose_and_fences(self):
        for text in (
            '{"a":1,"a":2}',
            '{"a":NaN}',
            '{"a":Infinity}',
            "```json\n{}\n```",
            "Here is {}",
        ):
            with self.subTest(text=text), self.assertRaises(SchemaError):
                load_json(text)
