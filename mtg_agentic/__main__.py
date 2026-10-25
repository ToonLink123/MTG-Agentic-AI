"""Small JSON CLI for bootstrap, discovery, and offline deck checks."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .oracle import DEFAULT_DATA_DIR, OracleStore
from .scryfall import ScryfallClient, ScryfallError
from .validator import validate_deck


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
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
    args = parser.parse_args(argv)

    try:
        result: Any
        if args.command == "validate":
            contents = args.deck.read_text(encoding="utf-8-sig")
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
            print(json.dumps(result, ensure_ascii=True, indent=2))
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
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except (ScryfallError, ValueError, OSError, EOFError, sqlite3.Error) as exc:
        error: dict[str, Any] = {"error": "scryfall" if isinstance(exc, ScryfallError) else "input"}
        error["detail"] = str(exc)
        if isinstance(exc, ScryfallError):
            error["status"] = exc.status
        print(json.dumps({"valid": False, "errors": [error]}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
