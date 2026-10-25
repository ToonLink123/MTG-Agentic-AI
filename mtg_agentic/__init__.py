"""Card discovery and deterministic Commander validation."""

from .brackets import BracketChecker, SpellbookClient
from .evaluation import evaluation_history, run_evaluation
from .forge import ForgeError, export_deck
from .forge_config import ForgeInstallation, OpponentPod
from .forge_metrics import summarize as summarize_simulation
from .forge_runner import ForgeSimulator, simulate
from .forge_support import ForgeSupport
from .oracle import OracleStore
from .pipeline import build_deck, discover, draft_deck
from .review import DeckReviewer
from .revision import revise_deck
from .rules import HashEmbedder, OllamaEmbedder, RulesStore, check_interaction
from .scryfall import CardNotFound, ScryfallClient, ScryfallError
from .sources import SourceCache, SourceError
from .spec import DeckSpec, intake
from .validator import DeckEntry, DeckValidator, validate_deck

__all__ = [
    "CardNotFound",
    "BracketChecker",
    "SpellbookClient",
    "DeckReviewer",
    "DeckEntry",
    "DeckSpec",
    "DeckValidator",
    "OracleStore",
    "RulesStore",
    "OllamaEmbedder",
    "HashEmbedder",
    "SourceCache",
    "SourceError",
    "ScryfallClient",
    "ScryfallError",
    "validate_deck",
    "build_deck",
    "discover",
    "draft_deck",
    "intake",
    "check_interaction",
    "run_evaluation",
    "evaluation_history",
    "ForgeError",
    "ForgeInstallation",
    "OpponentPod",
    "ForgeSimulator",
    "ForgeSupport",
    "export_deck",
    "simulate",
    "summarize_simulation",
    "revise_deck",
]
