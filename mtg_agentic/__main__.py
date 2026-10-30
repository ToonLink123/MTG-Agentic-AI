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

from . import bootstrap
from .agent import supports_tools
from .brackets import BracketChecker, SpellbookClient, bracket_rules
from .bulk_search import BulkSearch
from .card_graph import CARD_DIMENSION, index_cards, index_ready
from .council import MEMBER, ORCHESTRATOR, council_draft
from .discovery import POOL_MODES, CandidatePool
from .edit import Toolbox, edit_deck, polish_deck
from .embeddings import EMBEDDINGGEMMA_MODEL, EMBEDDINGGEMMA_REVISION
from .engine_map import build_engine_map
from .evaluation import evaluation_history, run_evaluation
from .forge import ForgeError
from .forge_cli import load_forge
from .forge_cli import register_commands as register_forge_commands
from .forge_cli import run_command as run_forge_command
from .forge_config import OpponentPod
from .forge_metrics import summarize
from .forge_runner import ForgeSimulator
from .interview import MAX_ROUNDS, interview_round
from .lands import generate_mana_base, is_land
from .llm import PROVIDERS, LlamaCppClient, LLMError, OllamaClient, OpenAICompatibleClient
from .mentions import find_cards, lookup, name_index
from .oracle import DEFAULT_DATA_DIR, OracleStore
from .pipeline import build_deck, discover, draft_deck, draft_from_request
from .playability import spec_for_deck, validate_playability
from .review import DeckReviewer, Judge
from .revision import revise_deck
from .rules import EVIDENCE as RULES_EVIDENCE
from .rules import HashEmbedder, OllamaEmbedder, RulesStore
from .rules_graph import GRAPH_VERSION
from .rules_index import DEFAULT_REPO as DEFAULT_RULES_INDEX
from .rules_index import PrebuiltIndex
from .schema import BRACKET3_MODES, SPEC_SCHEMA, SchemaError, load_json
from .scryfall import ScryfallClient, ScryfallError
from .sources import SourceCache, SourceError
from .spec import DeckSpec, intake
from .validator import validate_deck

EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Claude's 1M-token context holds a 1,000-card pool (about 130K tokens); local models
# usually run 32-64K contexts, so they keep the smaller pool.
POOL_LIMITS = {"claude": 1000}
DEFAULT_POOL_LIMIT = 350
POOL_HELP = "Most candidates (up to 1000); default 1000 with claude, 350 otherwise"
# Evidence items per rules question, half of them rules. Claude's context holds twice the
# default, which raised gold-rule recall on the retrieval benchmark from 87% to 96%.
EVIDENCE_LIMITS = {"claude": 36}
POLISH_HELP = (
    "Skip the polish pass: with a tool-capable model (claude), the orchestrator reviews a valid "
    "deck once more and swaps cards that do not serve the plan"
)
POOL_MODE_HELP = (
    "How discovery ranks the cards its searches find: popular (EDHREC rank only), relevant "
    "(fit to the deck only) or mixed (both, weighted; the default)"
)


def _model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=os.getenv("MTG_LLM_PROVIDER", "claude"),
    )
    parser.add_argument("--model", default=os.getenv("MTG_LLM_MODEL"))
    parser.add_argument("--llm-url", default=os.getenv("MTG_LLM_BASE_URL"))
    parser.add_argument("--llm-timeout", type=float, default=300)
    parser.add_argument("--context-size", type=int, default=65536)
    parser.add_argument("--effort", choices=EFFORTS, default="high")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="llama-cpp: disable reasoning in chat templates that support enable_thinking",
    )
    parser.add_argument(
        "--output", type=Path, help="Save JSON to a NEW file; existing files are never overwritten"
    )


def _council_models(args: argparse.Namespace):
    """Orchestrator and member clients; Claude defaults to Sonnet leading Haiku."""
    claude = args.provider == "claude"

    def make(model: str | None, effort: str):
        return _make_model(argparse.Namespace(**{**vars(args), "model": model, "effort": effort}))

    orchestrator = make(
        args.orchestrator_model or (ORCHESTRATOR[0] if claude else args.model),
        args.orchestrator_effort,
    )
    member = make(args.member_model or (MEMBER[0] if claude else args.model), args.member_effort)
    return orchestrator, member


def _bracket3_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bracket3-mode", choices=BRACKET3_MODES,
        help="Bracket 3 house rules: unrestricted ignores land-denial/two-card combo limits; "
        "limited-land-destruction keeps combo limits and bans destroying multiple lands "
        "controlled by one opponent (Scorched Ruins allowed). Default: saved mode or standard.",
    )


def _deck_mode(deck: dict[str, Any], bracket: int, override: str | None) -> str:
    mode = override if override is not None else deck.get(
        "bracket3_mode", deck.get("spec", {}).get("bracket3_mode", "standard")
    )
    bracket_rules(bracket, mode)
    return mode


def _make_model(args: argparse.Namespace):
    if args.provider == "ollama":
        return OllamaClient(
            args.model or "qwen2.5:7b",
            base_url=args.llm_url or "http://localhost:11434",
            timeout=args.llm_timeout,
            context_size=args.context_size,
        )
    if args.provider == "llama-cpp":
        return LlamaCppClient(
            *([args.model] if args.model else []),
            base_url=args.llm_url or "http://localhost:8080/v1",
            timeout=args.llm_timeout,
            context_size=args.context_size,
            api_key=os.getenv("MTG_LLM_API_KEY") or None,
            thinking=not args.no_thinking,
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
    "embeddinggemma": bootstrap.RULES_INDEX,
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
    parser.add_argument(
        "--rules-index",
        default=os.getenv("MTG_RULES_INDEX_REPO", DEFAULT_RULES_INDEX),
        help="Hugging Face dataset with prebuilt rule vectors, used when they match exactly; "
        "'none' always embeds locally",
    )


def _rules_store(args: argparse.Namespace, *, prebuilt: bool = True) -> RulesStore:
    repo = args.rules_index if prebuilt and args.rules_index not in {"", "none"} else None
    return RulesStore(
        _rules_path(args), _embedder(args), prebuilt=PrebuiltIndex(repo) if repo else None
    )


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


def _evidence(args: argparse.Namespace, default: int = RULES_EVIDENCE) -> int:
    return EVIDENCE_LIMITS.get(args.provider, default)


def _question_cards(args: argparse.Namespace, store: OracleStore, client: ScryfallClient) -> list[str]:
    """The cards a rules question is about: the names given, each found like a commander (so
    loose spellings and typos work), or else the cards the question names in plain words."""
    prefer = [args.commander] if args.commander else []
    if args.deck is not None:
        deck = _deck_json(args.deck)
        prefer += [deck["commander"], *deck["cards"]]
    index = name_index(store)
    if args.cards:
        cards = [lookup(name, store, client, prefer=prefer, index=index) for name in args.cards]
        missing = [name for name, card in zip(args.cards, cards) if card is None]
        if missing:
            raise ValueError("No card matches " + ", ".join(f"[[{name}]]" for name in missing))
        return list(dict.fromkeys(card["name"] for card in cards))
    found, unsettled = find_cards(args.question, store, prefer=prefer, index=index)
    if unsettled:
        raise ValueError("; ".join(
            f"\"{u['name']}\" could be " + " or ".join(u["options"]) for u in unsettled
        ) + ". Write the full name, or put it in [[double brackets]].")
    return found


def _toolbox(args, stack, store, client, reviewer) -> Toolbox:
    """What the drafter's and the polish pass's tools reach: local search, combos, rules."""
    sources = stack.enter_context(SourceCache(args.data_dir / "sources.sqlite3"))
    return Toolbox(
        store, client, reviewer.rules if reviewer is not None else None,
        SpellbookClient(sources, store=store), BulkSearch(store, client),
        evidence=_evidence(args, 12),
    )


def _polish(args, deck, pool, model, toolbox, reviewer) -> dict[str, Any]:
    """The polish pass on a valid deck, for a model with native tool use."""
    if args.no_polish or not deck.get("valid") or not supports_tools(model):
        return deck
    return polish_deck(deck, pool, model, tools=toolbox, reviewer=reviewer)


def _card_embedder(args, store):
    """The card index's embedder, when setup built the index and this Python can embed."""
    if bootstrap.embedding_problem() or not index_ready(store, bootstrap.card_index_identity()):
        return None
    from .embeddings import EmbeddingGemmaEmbedder

    try:
        return EmbeddingGemmaEmbedder(dimension=CARD_DIMENSION)
    except ValueError as exc:
        logging.getLogger("mtg_agentic").warning("card search: no meaning ranking (%s)", exc)
        return None


def _map_judge(args, stack, store, client, model):
    """The rules judge that checks an engine map's links, or None when setup has not built the
    rules index here (a map is still useful unchecked). It never downloads the rules itself:
    setup and the deck review keep them current."""
    path = _rules_path(args)
    if not path.is_file():
        logging.getLogger("mtg_agentic").warning("engine map: no rules index, so links are not checked")
        return None
    try:
        rules = stack.enter_context(_rules_store(args))
    except (ValueError, OSError) as exc:
        logging.getLogger("mtg_agentic").warning("engine map: links not checked (%s)", exc)
        return None
    if not rules.metadata.get("rule_count") or rules.metadata.get("graph_version") != GRAPH_VERSION:
        logging.getLogger("mtg_agentic").warning("engine map: the rules index is incomplete; run setup")
        return None
    return Judge(rules, model, store, client, evidence=_evidence(args))


def _reviewer(args, stack, store, client, model):
    sources = stack.enter_context(SourceCache(args.data_dir / "sources.sqlite3"))
    rules = stack.enter_context(_rules_store(args))
    rules.download_rules(sources)
    brackets = BracketChecker(store, client, sources, model)
    return DeckReviewer(brackets, rules, model, store, client, evidence=_evidence(args))


def _setup(args: argparse.Namespace) -> dict[str, Any]:
    """Download every missing piece the pipeline reads locally, in order, logging each stage."""
    done = (
        dict.fromkeys(bootstrap.STEPS, False) if args.refresh
        else bootstrap.status(args.data_dir, _rules_path(args).name)
    )
    steps: dict[str, str] = {}
    with (
        OracleStore(args.data_dir / "oracle.sqlite3") as store,
        ScryfallClient(args.data_dir / "responses.sqlite3") as client,
    ):
        def cards() -> str:
            info = client.download_oracle_cards(store, force=args.refresh)
            store.text_count("treasure")  # Builds the full-text index local search uses.
            return f"{info['card_count']:,} cards"

        def rules() -> str:
            with (
                SourceCache(args.data_dir / "sources.sqlite3") as sources,
                _rules_store(args) as index,
            ):
                return f"{index.download_rules(sources, refresh=args.refresh)['rule_count']} rules"

        def model() -> str:
            if args.embedding_backend != "embeddinggemma":
                return f"skipped: the {args.embedding_backend} backend needs no download"
            if problem := bootstrap.embedding_problem():
                return f"skipped: {problem}"
            if bootstrap.model_cached() and not args.refresh:
                return bootstrap.ALREADY
            bootstrap.fetch_model()
            return "EmbeddingGemma 2 downloaded"

        def card_index() -> str:
            if args.embedding_backend != "embeddinggemma":
                return f"skipped: the {args.embedding_backend} backend embeds no cards"
            if problem := bootstrap.embedding_problem():
                return f"skipped: {problem}"
            if index_ready(store, bootstrap.card_index_identity()) and not args.refresh:
                return bootstrap.ALREADY
            from .embeddings import EmbeddingGemmaEmbedder

            embedder = EmbeddingGemmaEmbedder(
                dimension=CARD_DIMENSION, device=args.embedding_device, batch_size=64,
            )
            return f"{index_cards(store, embedder):,} cards embedded"

        actions = {
            "Card data": cards,
            "Oracle tags": lambda: f"{client.download_oracle_tags(store, force=args.refresh)['tag_count']:,} tags",
            "Rulings": lambda: f"{client.download_rulings(store, force=args.refresh)['ruling_count']:,} rulings",
            "Commander Spellbook": lambda: "skipped: --skip-combos" if args.skip_combos else (
                f"{SpellbookClient(store=store).download_variants(store, force=args.refresh)['combo_count']:,} combos"
            ),
            "Rules index": rules,
            "Embedding model": model,
            "Card index": card_index,
        }
        for name in bootstrap.STEPS:
            # The model steps always check for themselves, to say why they skip.
            if done[name] and name not in {"Embedding model", "Card index"}:
                bootstrap.stage(name, bootstrap.ALREADY)
                steps[name] = bootstrap.ALREADY
                continue
            bootstrap.stage(name, "downloading")
            steps[name] = actions[name]()
            bootstrap.stage(name, steps[name])
    return {"ready": True, "steps": steps}


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
    bulk = commands.add_parser(
        "download-oracle",
        help="Download/index Oracle Cards, Oracle tags, rulings and Commander Spellbook combos once",
    )
    bulk.add_argument("--refresh", action="store_true")
    bulk.add_argument(
        "--skip-combos", action="store_true",
        help="Skip the Commander Spellbook export (about 700 MB, parsed as it downloads); "
        "combo checks then query its API",
    )
    setup_cmd = commands.add_parser(
        "setup",
        help="Download everything the pipeline reads locally, once: card data, Oracle tags, "
        "rulings, Commander Spellbook combos, the rules index and the embedding model "
        "(about 2.3 GB); later runs only fetch what is missing",
    )
    setup_cmd.add_argument("--refresh", action="store_true", help="Download fresh copies of everything")
    setup_cmd.add_argument("--skip-combos", action="store_true",
                           help="Skip the 700 MB Commander Spellbook export; combo checks use its API")
    _embedding_options(setup_cmd)
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
    mana_base.add_argument("--lands", type=int, choices=range(100), help="Override the spec land count (default 36 without a spec; 0 sets it from the curve)")
    mana_base.add_argument("--output", type=Path, help="Write the mana-base report to a NEW file")
    commands.add_parser("spec-schema", help="Print the fixed intake JSON schema")
    interview_cmd = commands.add_parser(
        "interview",
        help="One round of playstyle questions before intake; run again with the answers until "
        'the status is "ready"',
    )
    interview_cmd.add_argument("--commander", required=True)
    interview_cmd.add_argument("--request", required=True)
    interview_cmd.add_argument(
        "--answers", type=Path,
        help="JSON list of earlier answers: {question, answer, source: option|own|agents}",
    )
    interview_cmd.add_argument("--round", dest="round_number", type=int, default=1)
    interview_cmd.add_argument(
        "--max-rounds", type=int, choices=range(1, 21), default=MAX_ROUNDS, metavar="1-20",
        help=f"The last round must be ready (default {MAX_ROUNDS})",
    )
    _bracket3_options(interview_cmd)
    _model_options(interview_cmd)
    intake_cmd = commands.add_parser("intake", help="Ground preferences in fetched commander text")
    intake_cmd.add_argument("--commander", required=True)
    intake_cmd.add_argument("--request", required=True)
    intake_cmd.add_argument(
        "--interview", type=Path, help="The interview's answers, as for interview --answers"
    )
    _bracket3_options(intake_cmd)
    _model_options(intake_cmd)
    discover_cmd = commands.add_parser(
        "discover", help="Generate live queries and save a tagged pool"
    )
    discover_cmd.add_argument("--spec", required=True, type=Path)
    discover_cmd.add_argument("--pool-limit", type=int, help=POOL_HELP)
    discover_cmd.add_argument("--pool-mode", choices=POOL_MODES, default="mixed", help=POOL_MODE_HELP)
    _model_options(discover_cmd)
    map_cmd = commands.add_parser(
        "engine-map", help="Map the pool's resource loop for the brief; prints the mapped pool"
    )
    map_cmd.add_argument("--pool", required=True, type=Path)
    map_cmd.add_argument(
        "--no-combos", action="store_true", help="Skip the Commander Spellbook combo lookup"
    )
    map_cmd.add_argument(
        "--no-verify", action="store_true",
        help="Skip checking the map's central links against the rules before drafting",
    )
    _embedding_options(map_cmd)
    _model_options(map_cmd)
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
    draft_cmd.add_argument("--no-polish", action="store_true", help=POLISH_HELP)
    _embedding_options(draft_cmd)
    _model_options(draft_cmd)
    council_cmd = commands.add_parser(
        "council",
        help="Draft with an orchestrator and members that debate only the cards they disagree on",
    )
    council_cmd.add_argument("--pool", required=True, type=Path)
    council_cmd.add_argument("--members", type=int, choices=range(2, 7), default=3)
    council_cmd.add_argument("--repairs", type=int, choices=range(4), default=3)
    council_cmd.add_argument(
        "--orchestrator-model", help=f"Default: {ORCHESTRATOR[0]} for claude, else --model"
    )
    council_cmd.add_argument("--orchestrator-effort", choices=EFFORTS, default=ORCHESTRATOR[1])
    council_cmd.add_argument("--member-model", help=f"Default: {MEMBER[0]} for claude, else --model")
    council_cmd.add_argument("--member-effort", choices=EFFORTS, default=MEMBER[1])
    council_cmd.add_argument(
        "--skip-review", action="store_true",
        help="Explicitly skip bracket/rules safety review (not evaluation-ready)",
    )
    council_cmd.add_argument("--no-polish", action="store_true", help=POLISH_HELP)
    _embedding_options(council_cmd)
    _model_options(council_cmd)
    edit_cmd = commands.add_parser(
        "edit-deck",
        help="Ask the orchestrator about a drafted deck or for changes; it can use tools",
    )
    edit_cmd.add_argument("deck", type=Path)
    edit_cmd.add_argument(
        "--pool", type=Path, help="The deck's candidate pool; without it, only searches add cards"
    )
    edit_cmd.add_argument("--request", required=True, help="The player's message")
    edit_cmd.add_argument("--steps", type=int, choices=range(2, 31), default=10,
                          help="Most model steps, tool calls included")
    edit_cmd.add_argument(
        "--skip-review", action="store_true",
        help="Explicitly skip bracket/rules safety review of the changes",
    )
    _embedding_options(edit_cmd)
    _model_options(edit_cmd)
    build_cmd = commands.add_parser(
        "build", help="Draft, pass safety reviews, then complete at least 10 Forge games per pod"
    )
    build_cmd.add_argument("--commander", required=True)
    build_cmd.add_argument("--request", required=True)
    _bracket3_options(build_cmd)
    build_cmd.add_argument("--pool-limit", type=int, help=POOL_HELP)
    build_cmd.add_argument("--pool-mode", choices=POOL_MODES, default="mixed", help=POOL_MODE_HELP)
    build_cmd.add_argument("--repairs", type=int, choices=range(4), default=3)
    _simulation_options(build_cmd, required=True)
    _embedding_options(build_cmd)
    _model_options(build_cmd)
    download_rules = commands.add_parser(
        "download-rules", help="Download and embed current CR text"
    )
    download_rules.add_argument("--refresh", action="store_true")
    _embedding_options(download_rules)
    export_index = commands.add_parser(
        "export-rules-index", help="Embed the current rules locally and write files to publish"
    )
    export_index.add_argument("--output", type=Path, required=True, help="Directory to write")
    _embedding_options(export_index)
    for name in ("bracket-check", "review-deck"):
        cmd = commands.add_parser(name, help="Assess a JSON deck; review-deck also checks rules")
        cmd.add_argument("deck", type=Path)
        cmd.add_argument("--bracket", type=int, choices=range(1, 6))
        _bracket3_options(cmd)
        _model_options(cmd)
        _embedding_options(cmd)
    interaction = commands.add_parser("check-interaction", help="Require rule/ruling citations")
    interaction.add_argument(
        "cards", nargs="*",
        help="Card names, spelled loosely or misspelled; without any, the cards the question names",
    )
    interaction.add_argument("--question", required=True)
    interaction.add_argument("--commander", help="Settles a short name such as Jodah")
    interaction.add_argument("--deck", type=Path, help="A deck whose cards settle short names")
    _model_options(interaction)
    _embedding_options(interaction)
    evaluate = commands.add_parser("evaluate", help="Run all four fixed cases with safety reviews")
    evaluate.add_argument("--runs-dir", type=Path)
    evaluate.add_argument("--label", default="")
    evaluate.add_argument("--repairs", type=int, choices=range(4), default=3)
    evaluate.add_argument("--pool-limit", type=int, help=POOL_HELP)
    evaluate.add_argument("--pool-mode", choices=POOL_MODES, default="mixed", help=POOL_MODE_HELP)
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
    tui = commands.add_parser("tui", help="Dark terminal chat: pick a commander, describe a deck")
    tui.add_argument("--provider", choices=PROVIDERS, help="Default: MTG_LLM_PROVIDER or claude")
    tui.add_argument("--model", help="Default: MTG_LLM_MODEL or the provider's default")
    tui.add_argument("--pool-mode", choices=POOL_MODES, default="mixed",
                     help="The starting /pool: " + POOL_MODE_HELP)
    register_forge_commands(commands)
    args = parser.parse_args(argv)
    if getattr(args, "pool_limit", 0) is None:
        args.pool_limit = POOL_LIMITS.get(args.provider, DEFAULT_POOL_LIMIT)
    if args.command == "tui":
        try:
            from .tui import run_tui  # Loads the optional Textual package only when selected.
        except ModuleNotFoundError as exc:
            if exc.name != "textual" and not (exc.name or "").startswith("textual."):
                raise
            print(json.dumps({"valid": False, "errors": [
                {"error": "input", "detail": 'The TUI needs: python -m pip install -e ".[tui]"'}
            ]}))
            return 2
        return run_tui(args.data_dir, provider=args.provider, model=args.model,
                       pool=args.pool_mode)
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
        if args.command == "setup":
            _emit(_setup(args))
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
                _rules_store(args) as rules,
            ):
                _emit(rules.download_rules(sources, refresh=args.refresh))
            return 0
        if args.command == "export-rules-index":
            # Never prebuilt: a published index must come from embeddings computed here.
            with (
                SourceCache(args.data_dir / "sources.sqlite3") as sources,
                _rules_store(args, prebuilt=False) as rules,
            ):
                _emit(rules.export_index(sources, args.output))
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
                    mode = _deck_mode(deck, bracket, args.bracket3_mode)
                    result = BracketChecker(store, client, sources, model).check(
                        deck["cards"], deck["commander"], bracket, bracket3_mode=mode
                    )
                    _emit(result, args.output)
                    return 0 if result["verdict"] == "fits" else 1
                if args.command == "check-interaction":
                    canonical = _question_cards(args, store, client)
                    rules = stack.enter_context(_rules_store(args))
                    rules.download_rules(sources)
                    rules.ingest_deck(canonical, store, client)
                    result = rules.check_interaction(
                        canonical, args.question, model, limit=_evidence(args)
                    )
                    name, effort = getattr(model, "model", None), getattr(model, "effort", None)
                    result["answered_by"] = {
                        "provider": args.provider,
                        "model": name if isinstance(name, str) else None,
                        **({"effort": effort} if isinstance(effort, str) else {}),
                    }
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
                    mode = _deck_mode(deck, bracket, args.bracket3_mode)
                    validation = validate_deck(
                        deck["cards"],
                        deck["commander"],
                        store,
                        bracket=bracket,
                        game_changers=reviewer.game_changer_names,
                    )
                    playability = validate_playability(deck["cards"], spec_for_deck(deck, store, bracket=bracket, bracket3_mode=mode), store)
                    result = reviewer.check(deck["cards"], deck["commander"], bracket, bracket3_mode=mode) if validation["valid"] and playability["valid"] else {"status": "needs_revision", "errors": validation["errors"] + playability["errors"]}
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
                    search = BulkSearch(store, client)
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
                            search=search,
                            pool_mode=args.pool_mode,
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
                            "pool_mode": args.pool_mode,
                            "max_repairs": args.repairs,
                            "simulation": sim_config,
                        },
                        simulation_runner=simulation_runner,
                        reviewer=reviewer if simulation_runner else None,
                    )
                    result["valid"] = result["passed"]
                _emit(result, args.output)
                return 0 if result["valid"] else 1
        if args.command == "edit-deck":
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
                sources = stack.enter_context(SourceCache(args.data_dir / "sources.sqlite3"))
                # The orchestrator's rules tool needs the index even when the review is off.
                if args.skip_review:
                    reviewer = None
                    rules = stack.enter_context(_rules_store(args))
                    rules.download_rules(sources)
                else:
                    reviewer = _reviewer(args, stack, store, client, model)
                    rules = reviewer.rules
                deck = _deck_json(args.deck)
                pool = (
                    CandidatePool.from_dict(
                        load_json(args.pool.read_text(encoding="utf-8-sig")), store
                    )
                    if args.pool is not None
                    else CandidatePool(spec_for_deck(deck, store), [])
                )
                result = edit_deck(
                    deck, pool, model, args.request,
                    tools=Toolbox(store, client, rules, SpellbookClient(sources, store=store),
                                  BulkSearch(store, client), evidence=_evidence(args, 12)),
                    reviewer=reviewer, max_steps=args.steps,
                )
            _emit(result, args.output)
            return 1 if result.get("valid") is False else 0
        if args.command == "council":
            if args.output is not None and args.output.exists():
                raise ValueError("Output already exists; choose a new filename")
            orchestrator, member = _council_models(args)
            with (
                ExitStack() as stack,
                OracleStore(args.data_dir / "oracle.sqlite3") as store,
                ScryfallClient(args.data_dir / "responses.sqlite3") as client,
            ):
                for model in (member, orchestrator):
                    if hasattr(model, "report"):
                        stack.callback(model.report)
                if not store.ready:
                    raise ValueError("Run download-oracle before using the LLM pipeline")
                # The many parallel rules questions go to the cheaper member model.
                reviewer = None if args.skip_review else _reviewer(args, stack, store, client, member)
                pool = CandidatePool.from_dict(
                    load_json(args.pool.read_text(encoding="utf-8-sig")), store
                )
                toolbox = _toolbox(args, stack, store, client, reviewer)
                result = council_draft(
                    pool, orchestrator, member, client, store,
                    members=args.members, max_repairs=args.repairs, reviewer=reviewer,
                    toolbox=toolbox,
                )
                result = _polish(args, result, pool, orchestrator, toolbox, reviewer)
            _emit(result, args.output)
            return 1 if result.get("valid") is False else 0
        if args.command in {"interview", "intake", "discover", "engine-map", "draft", "build"}:
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
                if args.command == "interview":
                    leader = store.get_card(args.commander)
                    if leader is None:
                        raise ValueError("Commander is absent from the local card data")
                    answers = (
                        load_json(args.answers.read_text(encoding="utf-8-sig"))
                        if args.answers is not None else []
                    )
                    result = interview_round(
                        args.request, leader, answers, model, round_number=args.round_number,
                        max_rounds=args.max_rounds, bracket3_mode=args.bracket3_mode,
                    )
                elif args.command == "intake":
                    answers = (
                        load_json(args.interview.read_text(encoding="utf-8-sig"))
                        if args.interview is not None else []
                    )
                    result = intake(
                        args.request, args.commander, model, client, store,
                        bracket3_mode=args.bracket3_mode, interview=answers,
                    ).to_dict()
                elif args.command == "engine-map":
                    pool = CandidatePool.from_dict(
                        load_json(args.pool.read_text(encoding="utf-8-sig")), store
                    )
                    sources = stack.enter_context(SourceCache(args.data_dir / "sources.sqlite3"))
                    spellbook = None if args.no_combos else SpellbookClient(sources, store=store)
                    judge = None if args.no_verify else _map_judge(args, stack, store, client, model)
                    try:
                        pool = build_engine_map(
                            pool, model, store, spellbook=spellbook,
                            search=BulkSearch(store, client), judge=judge,
                        )
                        result = pool.to_dict()
                    except SchemaError as exc:
                        # Drafting still works without a map, so this is a warning, not an error.
                        result = {**pool.to_dict(), "valid": False, "map_errors": exc.errors}
                elif args.command == "discover":
                    spec = DeckSpec.from_dict(load_json(args.spec.read_text(encoding="utf-8-sig")))
                    result = discover(
                        spec, model, client, store, pool_limit=args.pool_limit,
                        search=BulkSearch(store, client), pool_mode=args.pool_mode,
                        embedder=_card_embedder(args, store),
                    ).to_dict()
                elif args.command == "draft":
                    pool = CandidatePool.from_dict(
                        load_json(args.pool.read_text(encoding="utf-8-sig")), store
                    )
                    toolbox = _toolbox(args, stack, store, client, reviewer)
                    result = draft_deck(
                        pool, model, client, store, max_repairs=args.repairs, reviewer=reviewer,
                        toolbox=toolbox,
                    )
                    result = _polish(args, result, pool, model, toolbox, reviewer)
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
                        bracket3_mode=args.bracket3_mode,
                        search=BulkSearch(store, client),
                        pool_mode=args.pool_mode,
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
                    result = {
                        **result,
                        "tags": client.download_oracle_tags(store, force=args.refresh),
                        "rulings": client.download_rulings(store, force=args.refresh),
                    }
                    if not args.skip_combos:
                        result["combos"] = SpellbookClient(store=store).download_variants(
                            store, force=args.refresh
                        )
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
