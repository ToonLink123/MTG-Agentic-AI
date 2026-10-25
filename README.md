# MTG-Agentic-AI

Scryfall-backed card discovery and deterministic Commander deck validation. Python
3.10+, no runtime dependencies, no API key, and no LLM involved in validation.

## Quick start

Run from the repository root (or install with `python -m pip install -e .`).
If your default Python is older than 3.10, use `uv run --no-project --python 3.11`
in place of `python` below.

```sh
# Download Oracle Cards and create the local SQLite index, once.
python -m mtg_agentic download-oracle

# Search, exact-then-fuzzy lookup, and rulings.
python -m mtg_agentic search "t:equipment id:w legal:commander"
python -m mtg_agentic card "Sram, Senior Edificer"
python -m mtg_agentic rulings <scryfall-card-id>

# The example intentionally fails: Lightning Bolt is outside Sram's identity.
python -m mtg_agentic validate examples/deck.json

# Refresh the snapshot after new sets, bans, or Game Changer updates.
python -m mtg_agentic download-oracle --refresh
```

All command results are JSON. Validation exits with `0` for valid, `1` for an
invalid deck, and `2` for an input/HTTP/storage failure. Use
`python -m mtg_agentic --data-dir /path/to/data ...` to change the cache location.
The default is `.cache/mtg-agentic/`, excluded from Git.

## Scryfall client

```python
from mtg_agentic import ScryfallClient

with ScryfallClient() as client:
    cards = client.search_cards("t:equipment id:w legal:commander")
    card = client.get_card("Sol Ring")
    rulings = client.get_rulings(card["id"])
```

`search_cards(query)` follows every page of `/cards/search`.
`get_card(name)` tries `/cards/named?exact=...`, using fuzzy only after a 404.
`get_rulings(card_id)` retrieves and paginates `/cards/:id/rulings`.

Card results contain **only** `id`, `name`, `mana_cost`, `type_line`,
`oracle_text`, `color_identity`, and `game_changer`. Multiple faces are combined,
so back-face text and costs are not lost. The ID allows subsequent ruling
lookups. Rulings contain only `source`, `published_at`, and `comment`.

Requests send explicit `User-Agent` and `Accept` headers. Every request is spaced
at least 100ms apart; search and name requests also observe Scryfall's current
500ms endpoint limits. HTTP 429 errors are surfaced, not automatically retried;
the client blocks subsequent network requests for at least 30 seconds or the
longer `Retry-After` value. Reuse one client in a single thread to share throttling.
Do not run multiple clients/processes to bypass rate limits.

Successful responses and 404s are persisted in SQLite for 24 hours by default.
Other failures are not cached. Empty searches return `[]`; unresolved or ambiguous
name lookups raise `CardNotFound`. Other HTTP/connection errors raise
`ScryfallError`. Search never silently returns partial results after a page fails.

## Local Oracle Cards

```python
from mtg_agentic import OracleStore, ScryfallClient

with OracleStore() as store, ScryfallClient() as client:
    metadata = client.download_oracle_cards(store)
    exact_card = store.get_card("Sol Ring")  # Local only; None if absent.
```

Bootstrap downloads the Oracle Cards bulk file and builds a trimmed, indexed
SQLite snapshot. Current gzipped JSONL exports and older JSON arrays are streamed,
not loaded entirely into memory. The downloaded file is retained next to the
index. Later bootstrap calls skip **all** network access when the index exists.
`force=True` (CLI `--refresh`) downloads a fresh snapshot. Imports are transactional:
bad or interrupted imports preserve the previous index.

You can also use `store.import_file("/path/to/oracle-cards.jsonl.gz")` to import an
existing download. `store.metadata` records the snapshot's update time and count.

Validation is only as current as this snapshot. Refresh after set releases or
Commander policy updates, or at least weekly. No hardcoded banned-card or Game
Changer list is maintained; both come from Scryfall's snapshot.

## Deck validation

Supply the commander **separately**, plus exactly 99 maindeck cards. Including
another copy of the commander in the maindeck is a singleton violation, not
silently deduplicated. JSON decks use this shape:

```json
{
  "commander": "Sram, Senior Edificer",
  "bracket": 3,
  "cards": {"Plains": 97, "Sol Ring": 1, "Lightning Bolt": 1}
}
```

Plain-text maindecks accept `1 Card Name`, `1x Card Name`, or bare card names,
with blank lines and whole-line `#`/`//` comments ignored. Repeated lines are
aggregated. Export-specific set/collector annotations and sideboards are not
parsed. For text files, pass `--commander "Sram, Senior Edificer"`.

```python
from mtg_agentic import DeckValidator, OracleStore, validate_deck
from mtg_agentic.validator import DeckEntry

with OracleStore() as store:
    report = validate_deck(
        {"Plains": 97, "Sol Ring": 1, "Lightning Bolt": 1},
        "Sram, Senior Edificer",
        store,
        bracket=3,
    )
    # Also accepts text, lists of text lines, or DeckEntry(name, quantity).
    report = DeckValidator(store).validate([DeckEntry("Plains", 99)], "Sram, Senior Edificer")
```

Checks:

- Exactly 100 cards including the commander.
- Real card names via exact, case-insensitive local lookups. **No fuzzy matching.**
- Singleton, including repeated/aliased names and the commander.
- Basic lands and Oracle-text duplicate exceptions, including unlimited copies
  (e.g. Hare Apparent) and bounded copies (Seven Dwarves and Nazgûl).
- Every card's color identity is a subset of the commander's.
- Every card, including the commander, has `legalities.commander == "legal"`.
- Commander eligibility, using the front face, explicit commander permissions,
  and Grist's special creature status.
- Game Changer count, including the commander and all copies. Brackets 1/2 allow
  zero, Bracket 3 allows three, and Brackets 4/5 have no count limit.

Reports are JSON-serializable dictionaries with `valid`, `errors`, `card_count`,
`commander`, `color_identity`, `bracket`, `game_changer_count`, `game_changers`,
and `oracle_updated_at`. Each error includes a stable `error` code and `detail`;
card-specific errors also include `card`, for example:

```json
{"error": "color_identity", "card": "Lightning Bolt", "detail": "R outside WU"}
```

Other error codes include `card_count`, `singleton`, `unknown_card`,
`commander_legality`, `commander_type`, `game_changers`, `invalid_entry`,
`invalid_decklist`, `invalid_commander`, `invalid_bracket`, `unsupported_commander`,
and `oracle_unavailable`. An empty local index fails closed, never validates a deck.

### Scope

This first version supports **one commander**. Partner/Background/Doctor's
companion pairings, chosen-color commanders, companions, sideboards, and house
rules are not supported. A list supplied as the commander is rejected.
Prismatic Piper and Faceless One return `unsupported_commander`.

The bracket check covers **Game Changer counts only**, not full bracket
certification: combo speed, extra turns, mass land denial, and playgroup intent
are outside this validator. `valid` means these implemented checks passed
against the recorded snapshot, not that all social bracket expectations were met.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use synthetic records, mocked HTTP, and a fake clock. They require no
network, no bulk download, and no third-party test libraries.

API references: [Scryfall](https://scryfall.com/docs/api),
[rate limits](https://scryfall.com/docs/api/rate-limits),
[bulk data](https://scryfall.com/docs/api/bulk-data).
Card data belongs to Wizards of the Coast and is provided by Scryfall.
