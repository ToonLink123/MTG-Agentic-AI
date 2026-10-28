"""Small JSON CLI for bootstrap, discovery, and offline deck checks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .brackets import BracketChecker
from .discovery import CandidatePool
from .embeddings import EMBEDDINGGEMMA_MODEL, EMBEDDINGGEMMA_REVISION
from .evaluation import evaluation_history, run_evaluation
from .forge import ForgeError
from .forge_cli import load_forge
from .forge_cli import register_commands as register_forge_commands
from .forge_cli import run_command as run_forge_command
from .forge_config import OpponentPod
from .forge_metrics import summarize
from .forge_runner import ForgeSimulator
from .lands import generate_mana_base, is_land
from .llm import LLMError, OllamaClient, OpenAICompatibleClient
from .oracle import DEFAULT_DATA_DIR, OracleStore
from .pipeline import build_deck, discover, draft_deck, draft_from_request
from .playability import spec_for_deck, validate_playability
from .review import DeckReviewer
from .revision import revise_deck
from .rules import HashEmbedder, OllamaEmbedder, RulesStore
from .schema import SPEC_SCHEMA, SchemaError, load_json
from .scryfall import ScryfallClient, ScryfallError
from .sources import SourceCache, SourceError
from .spec import DeckSpec, intake
from .validator import validate_deck


def _model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai-compatible", "claude"),
        default=os.getenv("MTG_LLM_PROVIDER", "ollama"),
    )
    parser.add_argument("--model", default=os.getenv("MTG_LLM_MODEL"))
    parser.add_argument("--llm-url", default=os.getenv("MTG_LLM_BASE_URL"))
    parser.add_argument("--llm-timeout", type=float, default=300)
    parser.add_argument("--context-size", type=int, default=65536)
    parser.add_argument(
        "--effort", choices=("low", "medium", "high", "xhigh", "max"), default="high"
    )
    parser.add_argument(
        "--output", type=Path, help="Save JSON to a NEW file; existing files are never overwritten"
    )


def _make_model(args: argparse.Namespace):
    if args.provider == "ollama":
        return OllamaClient(
            args.model or "qwen2.5:7b",
            base_url=args.llm_url or "http://localhost:11434",
            timeout=args.llm_timeout,
            context_size=args.context_size,
        )
    if args.provider == "claude":
        from .claude import ClaudeClient  # Loads the optional Anthropic SDK only when selected.

        return ClaudeClient(
            args.model or "claude-haiku-5-5",
            api_key=_claude_api_key(),
            timeout=args.llm_timeout,
            effort=args.effort,
            progress=None if args.quiet else sys.stderr,
        )
    if args.provider != "openai-compatible":
        raise ValueError("Unknown MTG_LLM_PROVIDER")
    if not args.model:
        raise ValueError("Select --model or MTG_LLM_MODEL for the API provider")
    return OpenAICompatibleClient(
        args.model,
        api_key=os.getenv("MTG_LLM_API_KEY", ""),
        base_url=args.llm_url or "https://api.openai.com/v1",
        timeout=args.llm_timeout,
    )


def _claude_api_key(env_file: Path = Path(".env")) -> str:
    """CLAUDE_API from the environment, else from a local, git-ignored .env file."""
    if os.getenv("CLAUDE_API"):
        return os.environ["CLAUDE_API"]
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ").strip() == "CLAUDE_API":
                return value.strip().strip("'\"")
    return ""


RULES_INDEXES = {
    "embeddinggemma": "rules-embeddinggemma-2.sqlite3",
    "ollama": "rules.sqlite3",
    "hash": "rules-hash.sqlite3",
}


def _embedding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-backend",
        choices=tuple(RULES_INDEXES),
        default="embeddinggemma",
        help="Rules embeddings: local Hugging Face EmbeddingGemma 2 (default), Ollama, or hash",
    )
    parser.add_argument(
        "--embedding-model",
        help=f"Default: {EMBEDDINGGEMMA_MODEL} (pinned revision) or nomic-embed-text for Ollama",
    )
    parser.add_argument("--embedding-url", default="http://localhost:11434")
    parser.add_argument("--embedding-device", help="embeddinggemma: cuda or cpu (default: auto)")


def _embedder(args: argparse.Namespace):
    if args.embedding_backend == "hash":
        return HashEmbedder()
    if args.embedding_backend == "ollama":
        return OllamaEmbedder(args.embedding_model or "nomic-embed-text", base_url=args.embedding_url)
    from .embeddings import EmbeddingGemmaEmbedder  # Imports torch/transformers only when used.

    model = args.embedding_model or EMBEDDINGGEMMA_MODEL
    return EmbeddingGemmaEmbedder(
        model,
        revision=EMBEDDINGGEMMA_REVISION if model == EMBEDDINGGEMMA_MODEL else None,
        device=args.embedding_device,
    )


def _rules_path(args: argparse.Namespace) -> Path:
    # Indexes are tied to their embedding model, so every backend keeps its own database.
    return args.data_dir / RULES_INDEXES[args.embedding_backend]


def _reviewer(args, stack, store, client, model):
    sources = stack.enter_context(SourceCache(args.data_dir / "sources.sqlite3"))
    rules = stack.enter_context(RulesStore(_rules_path(args), _embedder(args)))
    rules.download_rules(sources)
    brackets = BracketChecker(store, client, sources, model)
    return DeckReviewer(brackets, rules, model, store, client)


def _simulation_options(parser, *, required=False, games=10):
    parser.add_argument("--forge-home", type=Path, required=required)
    parser.add_argument(
        "--pod",
        type=Path,
        action="append",
        required=required,
        help="Pinned pod config; repeat for up to three comparison pods",
    )
    parser.add_argument("--decks-dir", type=Path)
    parser.add_argument("--java")
    parser.add_argument("--known-issues", type=Path)
    parser.add_argument("--games", type=int, default=games)
    parser.add_argument("--simulation-timeout", type=float, default=120)
    parser.add_argument("--startup-grace", type=float, default=30)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--logs-dir", type=Path)


def _simulation(args, store):
    if not args.pod or not 1 <= len(args.pod) <= 3:
        raise ValueError("Simulation requires one to three pinned --pod configs")
    installation, support = load_forge(args)
    pods = [OpponentPod.load(p, installation) for p in args.pod]
    if len({p.fingerprint for p in pods}) != len(pods):
        raise ValueError("Select distinct pods")
    simulator = ForgeSimulator(
        installation,
        store,
        support,
        args.logs_dir or args.data_dir / "simulations",
        timeout=args.simulation_timeout,
        startup_grace=args.startup_grace,
    )
    return simulator, pods


def _deck_json(path: Path) -> dict[str, Any]:
    data = load_json(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("commander"), str)
        or not isinstance(data.get("cards"), dict)
    ):
        raise ValueError("Expected deck JSON with commander and a cards mapping")
    return data


def _emit(result: Any, output: Path | None = None) -> None:
    serialized = json.dumps(result, ensure_ascii=True, indent=2)
    if output is None:
        print(serialized)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(serialized + "\n")
    summary: dict[str, Any] = {"output": str(output)}
    if isinstance(result, dict):
        if "valid" in result:
            summary["valid"] = result["valid"]
        if isinstance(result.get("cards"), list):
            summary["candidate_count"] = len(result["cards"])
    print(json.dumps(summary))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--quiet", action="store_true", help="No live progress on stderr")
    commands = parser.add_subparsers(dest="command", required=True)
    bulk = commands.add_parser("download-oracle", help="Download/index Oracle Cards once")
    bulk.add_argument("--refresh", action="store_true")
    search = commands.add_parser("search", help="Search Scryfall, with all pages trimmed")
    search.add_argument("query")
    card = commands.add_parser("card", help="Exact name lookup, then fuzzy fallback")
    card.add_argument("name")
    rulings = commands.add_parser("rulings")
    rulings.add_argument("card_id")
    validate = commands.add_parser("validate", help="Validate locally, with no HTTP calls")
    validate.add_argument("deck", type=Path, help="JSON deck or plain-text maindeck")
    validate.add_argument("--commander", help="Required for text decks")
    validate.add_argument("--bracket", type=int, choices=range(1, 6))
    validate.add_argument("--playability", action="store_true", help="Also check mana and roles (automatic for JSON artifacts containing a spec)")
    mana_base = commands.add_parser(
        "mana-base", help="Generate only lands from a finished spell list, without an LLM or HTTP"
    )
    mana_base.add_argument("deck", type=Path, help="JSON with commander, cards, and optional spec")
    mana_base.add_argument("--lands", type=int, choices=range(100), help="Override the spec land count (default 36 without a spec)")
    mana_base.add_argument("--output", type=Path, help="Write the mana-base report to a NEW file")
    commands.add_parser("spec-schema", help="Print the fixed intake JSON schema")
    intake_cmd = commands.add_parser("intake", help="Ground preferences in fetched commander text")
    intake_cmd.add_argument("--commander", required=True)
    intake_cmd.add_argument("--request", required=True)
    _model_options(intake_cmd)
    discover_cmd = commands.add_parser(
        "discover", help="Generate live queries and save a tagged pool"
    )
    discover_cmd.add_argument("--spec", required=True, type=Path)
    discover_cmd.add_argument("--pool-limit", type=int, default=350)
    _model_options(discover_cmd)
    draft_cmd = commands.add_parser(
        "draft", help="Choose spells, calculate lands, validate and repair"
    )
    draft_cmd.add_argument("--pool", required=True, type=Path)
    draft_cmd.add_argument("--repairs", type=int, choices=range(4), default=3)
    draft_cmd.add_argument(
        "--skip-review",
        action="store_true",
        help="Explicitly skip bracket/rules safety review (not evaluation-ready)",
    )
    _embedding_options(draft_cmd)
    _model_options(draft_cmd)
    build_cmd = commands.add_parser(
        "build", help="Draft, pass safety reviews, then complete at least 10 Forge games per pod"
    )
    build_cmd.add_argument("--commander", required=True)
    build_cmd.add_argument("--request", required=True)
    build_cmd.add_argument("--pool-limit", type=int, default=350)
    build_cmd.add_argument("--repairs", type=int, choices=range(4), default=3)
    _simulation_options(build_cmd, required=True)
    _embedding_options(build_cmd)
    _model_options(build_cmd)
    download_rules = commands.add_parser(
        "download-rules", help="Download and embed current CR text"
    )
    download_rules.add_argument("--refresh", action="store_true")
    _embedding_options(download_rules)
    for name in ("bracket-check", "review-deck"):
        cmd = commands.add_parser(name, help="Assess a JSON deck; review-deck also checks rules")
        cmd.add_argument("deck", type=Path)
        cmd.add_argument("--bracket", type=int, choices=range(1, 6))
        _model_options(cmd)
        _embedding_options(cmd)
    interaction = commands.add_parser("check-interaction", help="Require rule/ruling citations")
    interaction.add_argument("cards", nargs="+")
    interaction.add_argument("--question", required=True)
    _model_options(interaction)
    _embedding_options(interaction)
    evaluate = commands.add_parser("evaluate", help="Run all four fixed cases with safety reviews")
    evaluate.add_argument("--runs-dir", type=Path)
    evaluate.add_argument("--label", default="")
    evaluate.add_argument("--repairs", type=int, choices=range(4), default=3)
    evaluate.add_argument("--pool-limit", type=int, default=350)
    _model_options(evaluate)
    _embedding_options(evaluate)
    _simulation_options(evaluate)
    revise = commands.add_parser("revise", help="Bounded, safety-gated fixed-pod revisions")
    revise.add_argument("deck", type=Path)
    revise.add_argument("--pool", type=Path, required=True)
    revise.add_argument("--runs-dir", type=Path)
    revise.add_argument("--iterations", type=int, choices=range(1, 11), default=3)
    revise.add_argument("--plateau", type=int, default=2)
    _model_options(revise)
    _embedding_options(revise)
    _simulation_options(revise, required=True, games=50)
    history = commands.add_parser("evaluation-history")
    history.add_argument("--runs-dir", type=Path)
    register_forge_commands(commands)
    args = parser.parse_args(argv)
    if args.command in {"evaluate", "evaluation-history"} and args.runs_dir is None:
        args.runs_dir = args.data_dir / "evaluations"
    if args.command == "revise" and args.runs_dir is None:
        args.runs_dir = args.data_dir / "revisions"
    # Stage updates go to stderr so stdout stays clean JSON.
    progress = logging.StreamHandler(sys.stderr)
    progress.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("mtg_agentic")
    if not args.quiet:
        logger.setLevel(logging.INFO)
        logger.addHandler(progress)

    try:
        result: Any
        if args.command.startswith("forge-"):
            if args.output is not None and args.output.exists():
                raise ValueError("Output already exists; choose a new filename")
            result, code = run_forge_command(args)
            _emit(result, args.output)
            return code
        if args.command == "spec-schema":
            _emit(SPEC_SCHEMA)
            return 0
        if args.command == "mana-base":
            if args.output is not None and args.output.exists():
                raise ValueError("Output already exists; choose a new filename")
            with OracleStore(args.data_dir / "oracle.sqlite3") as store:
                if not store.ready:
                    raise ValueError("Run download-oracle first")
                deck = _deck_json(args.deck)
                data = spec_for_deck(deck, store).to_dict()
                if args.lands is not None or "spec" not in deck:
                    data["role_targets"]["lands"] = args.lands if args.lands is not None else 36
                spells, land_names = {}, []
                for name, quantity in deck["cards"].items():
                    card = store.get_card(name)
                    if card is None or type(quantity) is not int or quantity <= 0:
                        raise ValueError("Deck cards must have known names and positive integer quantities")
                    if is_land(card):
                        land_names.append(card["name"])
                    else:
                        spells[card["name"]] = spells.get(card["name"], 0) + quantity
                result = generate_mana_base(
                    spells, DeckSpec.from_dict(data), store, candidate_names=land_names
                )
                _emit(result, args.output)
                return 0 if result["valid"] else 1
        if args.command == "evaluation-history":
            _emit(evaluation_history(args.runs_dir))
            return 0
        if args.command == "download-rules":
            with (
                SourceCache(args.data_dir / "sources.sqlite3") as sources,
                RulesStore(_rules_path(args), _embedder(args)) as rules,
            ):
                _emit(rules.download_rules(sources, refresh=args.refresh))
            return 0
        if args.command in {
            "bracket-check",
            "review-deck",
            "check-interaction",
            "evaluate",
            "revise",
        }:
            if args.output is not None and args.output.exists():
                raise ValueError("Output already exists; choose a new filename")
            model = _make_model(args)
            with ExitStack() as stack:
                if hasattr(model, "report"):
                    stack.callback(model.report)
                store = stack.enter_context(OracleStore(args.data_dir / "oracle.sqlite3"))
                client = stack.enter_context(ScryfallClient(args.data_dir / "responses.sqlite3"))
                if not store.ready:
                    raise ValueError("Run download-oracle first")
                sources = stack.enter_context(SourceCache(args.data_dir / "sources.sqlite3"))
                if args.command == "bracket-check":
                    deck = _deck_json(args.deck)
                    bracket = args.bracket if args.bracket is not None else deck.get("bracket", 3)
                    result = BracketChecker(store, client, sources, model).check(
                        deck["cards"], deck["commander"], bracket
                    )
                    _emit(result, args.output)
                    return 0 if result["verdict"] == "fits" else 1
                if args.command == "check-interaction":
                    rules = stack.enter_context(RulesStore(_rules_path(args), _embedder(args)))
                    rules.download_rules(sources)
                    rules.ingest_deck(args.cards, store, client)
                    canonical = [store.get_card(n)["name"] for n in args.cards]
                    result = rules.check_interaction(canonical, args.question, model)
                    _emit(result, args.output)
                    return 0 if result["status"] == "supported" else 1
                reviewer = _reviewer(args, stack, store, client, model)
                if args.command == "revise":
                    simulator, pods = _simulation(args, store)
                    pool = CandidatePool.from_dict(
                        load_json(args.pool.read_text(encoding="utf-8-sig")), store
                    )
                    result = revise_deck(
                        _deck_json(args.deck),
                        pool,
                        model,
                        store,
                        reviewer,
                        simulator,
                        pods,
                        args.runs_dir,
                        games=args.games,
                        max_iterations=args.iterations,
                        plateau_limit=args.plateau,
                        seed=args.seed,
                    )
                    _emit(result, args.output)
                    return (
                        1
                        if result["stop_reason"]
                        in {"error", "cancelled", "initial_gates_failed", "simulation_unreliable"}
                        or not result["gate_invariant_passed"]
                        else 0
                    )
                if args.command == "review-deck":
                    deck = _deck_json(args.deck)
                    bracket = args.bracket if args.bracket is not None else deck.get("bracket", 3)
                    validation = validate_deck(
                        deck["cards"],
                        deck["commander"],
                        store,
                        bracket=bracket,
                        game_changers=reviewer.game_changer_names,
                    )
                    playability = validate_playability(deck["cards"], spec_for_deck(deck, store, bracket=bracket), store)
                    result = reviewer.check(deck["cards"], deck["commander"], bracket) if validation["valid"] and playability["valid"] else {"status": "needs_revision", "errors": validation["errors"] + playability["errors"]}
                    result["validation"] = validation
                    result["playability"] = playability
                    result["valid"] = result["status"] == "passed" and validation["valid"] and playability["valid"]
                else:
                    simulation_runner = None
                    sim_config = None
                    if args.forge_home is not None:
                        simulator, pods = _simulation(args, store)
                        sim_config = {
                            "installation": simulator.installation.provenance(),
                            "pods": [p.provenance() for p in pods],
                            "games": args.games,
                            "seed": args.seed,
                            "timeout": args.simulation_timeout,
                            "startup_grace": args.startup_grace,
                        }

                        def simulation_runner(deck):
                            results = []
                            for pod in pods:
                                raw = simulator.simulate(deck, pod, args.games, seed=args.seed)
                                results.append(
                                    {**summarize(raw), "artifact_dir": raw["artifact_dir"]}
                                )
                            return results
                    elif args.pod:
                        raise ValueError("--pod requires --forge-home")
                    result = run_evaluation(
                        lambda case: draft_from_request(
                            case["request"],
                            case["commander"],
                            model,
                            client,
                            store,
                            pool_limit=args.pool_limit,
                            max_repairs=args.repairs,
                            reviewer=reviewer,
                        ),
                        store,
                        args.runs_dir,
                        label=args.label,
                        configuration={
                            "provider": args.provider,
                            "model": model.model,
                            "embedding": reviewer.rules.metadata.get("embedding"),
                            "context_size": getattr(model, "context_size", None),
                            "pool_limit": args.pool_limit,
                            "max_repairs": args.repairs,
                            "simulation": sim_config,
                        },
                        simulation_runner=simulation_runner,
                        reviewer=reviewer if simulation_runner else None,
                    )
                    result["valid"] = result["passed"]
                _emit(result, args.output)
                return 0 if result["valid"] else 1
        if args.command in {"intake", "discover", "draft", "build"}:
            if args.output is not None and args.output.exists():
                raise ValueError("Output already exists; choose a new filename")
            model = _make_model(args)
            with (
                ExitStack() as stack,
                OracleStore(args.data_dir / "oracle.sqlite3") as store,
                ScryfallClient(args.data_dir / "responses.sqlite3") as client,
            ):
                if hasattr(model, "report"):
                    stack.callback(model.report)
                if not store.ready:
                    raise ValueError("Run download-oracle before using the LLM pipeline")
                reviewer = None
                simulator, pods = (None, [])
                if args.command == "build":
                    if args.games < 10:
                        raise ValueError("A build requires at least 10 games per opponent pod")
                    simulator, pods = _simulation(args, store)
                if args.command == "build" or (args.command == "draft" and not args.skip_review):
                    reviewer = _reviewer(args, stack, store, client, model)
                if args.command == "intake":
                    result = intake(args.request, args.commander, model, client, store).to_dict()
                elif args.command == "discover":
                    spec = DeckSpec.from_dict(load_json(args.spec.read_text(encoding="utf-8-sig")))
                    result = discover(
                        spec, model, client, store, pool_limit=args.pool_limit
                    ).to_dict()
                elif args.command == "draft":
                    pool = CandidatePool.from_dict(
                        load_json(args.pool.read_text(encoding="utf-8-sig")), store
                    )
                    result = draft_deck(
                        pool, model, client, store, max_repairs=args.repairs, reviewer=reviewer
                    )
                else:
                    result = build_deck(
                        args.request,
                        args.commander,
                        model,
                        client,
                        store,
                        pool_limit=args.pool_limit,
                        max_repairs=args.repairs,
                        reviewer=reviewer,
                        simulator=simulator,
                        pods=pods,
                        games=args.games,
                        seed=args.seed,
                    )
            _emit(result, args.output)
            return 1 if result.get("valid") is False else 0
        if args.command == "validate":
            contents = args.deck.read_text(encoding="utf-8-sig")
            deck = None
            if args.deck.suffix.lower() == ".json":
                deck = json.loads(contents)
                if not isinstance(deck, dict):
                    raise ValueError("Deck JSON must be an object with commander and cards")
                commander = args.commander if args.commander is not None else deck.get("commander")
                bracket = args.bracket if args.bracket is not None else deck.get("bracket", 3)
                decklist = deck.get("cards", {})
            else:
                if not args.commander:
                    raise ValueError("--commander is required for plain-text decks")
                commander = args.commander
                bracket = args.bracket if args.bracket is not None else 3
                decklist = contents
            with OracleStore(args.data_dir / "oracle.sqlite3") as store:
                result = validate_deck(decklist, commander, store, bracket=bracket)
                result["scope"] = "Commander legality only, not playability or safety approval"
                if result["valid"] and (args.playability or (deck and "spec" in deck)):
                    actual = {"commander": result["commander"], "bracket": bracket, "cards": result["cards"]}
                    if deck and "spec" in deck:
                        actual["spec"] = deck["spec"]
                    playability = validate_playability(actual["cards"], spec_for_deck(actual, store, bracket=bracket), store)
                    result["playability"] = playability
                    result["errors"].extend(playability["errors"])
                    result["valid"] = not result["errors"]
                    result["scope"] = "Commander legality and playability; safety review is separate"
            _emit(result)
            return 0 if result["valid"] else 1
        with ScryfallClient(args.data_dir / "responses.sqlite3") as client:
            if args.command == "download-oracle":
                with OracleStore(args.data_dir / "oracle.sqlite3") as store:
                    result = client.download_oracle_cards(store, force=args.refresh)
            elif args.command == "search":
                result = client.search_cards(args.query)
            elif args.command == "card":
                result = client.get_card(args.name)
            else:
                result = client.get_rulings(args.card_id)
        _emit(result)
        return 0
    except SchemaError as exc:
        print(json.dumps({"valid": False, "errors": exc.errors}))
        return 2
    except LLMError as exc:
        print(json.dumps({"valid": False, "errors": [{"error": "llm", "detail": str(exc)}]}))
        return 2
    except (
        ForgeError,
        SourceError,
        ScryfallError,
        ValueError,
        OSError,
        EOFError,
        sqlite3.Error,
    ) as exc:
        error: dict[str, Any] = {"error": "scryfall" if isinstance(exc, ScryfallError) else "input"}
        error["detail"] = str(exc)
        if isinstance(exc, ScryfallError):
            error["status"] = exc.status
        print(json.dumps({"valid": False, "errors": [error]}))
        return 2
    finally:
        logger.removeHandler(progress)


if __name__ == "__main__":
    sys.exit(main())
