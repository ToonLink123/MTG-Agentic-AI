# MTG-Agentic-AI

Scryfall-backed discovery, local-LLM deck drafting, and deterministic Commander
validation, with bracket review and citation-required interaction checks.
Optional local Forge simulation adds measured outcomes and gated revisions.
Python 3.10+, no runtime dependencies. Ollama runs locally without an API key;
the legality validator and land allocation never use an LLM.

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

## Intake -> spec -> discovery -> draft

This build pipeline stops at a reviewed draft. Forge runs are explicit, separate
commands, described below.

### Set up free local inference

Install [Ollama for Windows](https://ollama.com/download/windows), then download a
model once:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Keep Ollama running. The desktop app normally starts its server; otherwise run
`ollama serve` in a separate terminal. Models and long contexts require significant
RAM/VRAM, but no paid API account. Nothing downloads or installs a model automatically.

From the repository root, run these stages (replace `python` with
`uv run --no-project --python 3.11 python` if needed):

```powershell
python -m mtg_agentic download-oracle

# Optional: inspect the exact intake contract before making any model calls.
python -m mtg_agentic spec-schema

python -m mtg_agentic intake --commander "Jodah, Archmage Eternal" --request "Bracket 3 Myojin tribal with divinity counters, proliferate, and midrange value" --output .cache/jodah-spec.json

python -m mtg_agentic discover --spec .cache/jodah-spec.json --output .cache/jodah-pool.json

python -m mtg_agentic draft --pool .cache/jodah-pool.json --output .cache/jodah-deck.json

python -m mtg_agentic validate .cache/jodah-deck.json
```

To skip intake while testing discovery, use the checked-in
`examples/jodah-spec.json` as `--spec`. For the complete pipeline in one command:

```powershell
python -m mtg_agentic build --commander "Jodah, Archmage Eternal" --request "Bracket 3 Myojin tribal with divinity counters and proliferate" --output .cache/jodah-build.json
```

`--output` writes UTF-8 JSON to a **new** file and prints a small summary. Existing
files are never overwritten. Choose a new filename for another run. Without
`--output`, the complete JSON goes to stdout. Avoid PowerShell 5.1 redirection for
artifacts because it can write UTF-16; use `--output`.

Model flags, available on each LLM command:

- `--model qwen2.5:7b`, default `MTG_LLM_MODEL` or `qwen2.5:7b`.
- `--llm-url http://localhost:11434`, or `MTG_LLM_BASE_URL`.
- `--context-size 65536`, the default requested Ollama context.
- `--llm-timeout 300`, seconds per call.
- `--pool-limit 350` on `discover`/`build`, hard maximum 400.
- `--repairs 3` on `draft`/`build`, from 0 to 3.

On a smaller machine, try `--context-size 16384` and `--pool-limit 100` on the
relevant commands. A smaller pool can reduce theme/role coverage. A conservative
prompt-size estimate rejects oversized prompts before Ollama can silently
truncate them; increase context or rebuild a smaller pool if that happens.
Long draft calls can take minutes on CPU.

### Data contracts and safety checks

`SPEC_SCHEMA` in `mtg_agentic/schema.py` is the fixed JSON contract. It requires
exactly the requested fields: commander, color_identity, bracket, themes,
mechanics, win_conditions, playstyle, must_include, exclude, and role_targets.
Color identity uses WUBRG order, or `C` for colorless. All five role targets are
required integers from 0 to 99. Missing/extra fields, wrong types, duplicate
keys, invalid identities, and conflicting constraints are rejected in code.

Intake fetches the commander's actual trimmed Oracle text with an exact
`get_card` lookup before prompting. The response cannot change the commander or
invent its color identity. Required/excluded names must resolve locally.

Query planning is a second model call. Every query is grouped, then code appends
`f:commander id<=<identity>`, including fallback queries. All syntax and otags go
to the live API, using the existing cache and rate limits. Query-derived roles
are approximate categories, not a claim that every matching card performs that
role equally well.

Explicit tribal/counter/proliferate requests get deterministic query anchors,
even if the model omits them. Empty theme subtype queries can fall back to a
card-name search, and empty mechanic otags to Oracle text. This matters for
Jodah: **Myojin is a name family, not a creature subtype**. `t:myojin` has no
matches, while `myojin` finds the actual Spirit creatures. Likewise, an empty
`otag:proliferate` falls back to `o:"proliferate"`. Pool diagnostics record both
the original and fallback queries instead of hiding empty searches.

Discovery checks every result against local legality/color rules, removes the
commander and exclusions, merges role tags, and caps the pool at 350 by default.
Required cards and basic-land options are reserved. Query results are selected
round-robin so broad searches do not crowd out narrower themes. Each query is
bounded to 500 results by default, sorted by EDHREC; diagnostics disclose when
that limit is reached. Saved pools contain only trimmed fields plus role tags
and are rechecked against the local store when loaded.

The model selects `99 - role_targets.lands` **nonland** cards. Code supplies the
commander and land base. It then checks:

1. Strict draft JSON shape and exact commander.
2. Every nonland is in the pool or passes exact `get_card(..., fuzzy=False)`.
   A different/fuzzy name is rejected, never silently substituted. Exact new
   cards are indexed locally with rule fields before validation. Outside-pool
   lookups are capped at 12 for the entire loop and provide no role credit.
3. All existing deck-validator checks.
4. Required/excluded cards and minimum role counts. A card with several tags can
   count toward several roles; the land count is exact.

Errors go back to the model as JSON, with the previous draft, for up to **three
repairs after the initial draft**. Exhaustion returns `valid: false`, the last
deck, and all errors, not an unverified success. Drafts include attempt history,
role counts, query diagnostics, verification records, and land-base details.
They can also be passed directly to the existing `validate` command.

CLI `draft`/`build` now also run bracket and rules safety reviews by default.
Uncertain, mismatched, unsupported or contradicted findings enter the repair loop.
`--skip-review` explicitly opts out and leaves `review.status: "not_run"` and
`review_passed: false`. First-time rules embedding and interaction calls can take
several minutes on CPU. `validate` remains a narrower offline snapshot check; it
does not rerun safety review. Library functions retain injectable, network-free
defaults: pass a `DeckReviewer` to enable reviews, as shown below.

### Deterministic land base

Code counts colored/colorless mana pips, weights mana-value-2-or-less spells
three times and mana-value-3/4 spells twice, includes the commander, and splits
hybrid demand. Largest-remainder allocation sets color-source targets. Required
lands are pinned; eligible multicolor lands address source deficits, and basic
lands fill the remaining slots. Nonbasics remain singleton, exclusions apply,
and optional Game Changer lands respect the remaining bracket allowance.

The report contains weighted pips, source targets, potential sources, and
warnings. It uses front-face costs and available pool lands. It is a simple
heuristic, **not an exact Frank Karsten probability model**, an optimized mana
base, or a simulation. Fetch/conditional lands are only potential sources;
untapped timing, all activation restrictions, snow requirements, and affordability
are not modeled. Check the warnings and review the resulting mana base.

### Optional API-key provider

Ollama is always the default. To explicitly use a compatible hosted API later:

```powershell
# Set MTG_LLM_API_KEY securely in your environment, not in a checked-in file.
python -m mtg_agentic build --provider openai-compatible --model YOUR_MODEL --commander "Jodah, Archmage Eternal" --request "Myojin tribal"
```

Use `--llm-url https://your-provider.example/v1` for another compatible endpoint.
It must support Chat Completions with JSON-schema `response_format`. You can also
set `MTG_LLM_PROVIDER=openai-compatible`. An explicit model and key are required;
the app never silently falls back to a paid provider. Keys are not stored in
specs, pools, drafts, or cache files. Non-local model endpoints require HTTPS.

### Python API

```python
from mtg_agentic import OracleStore, ScryfallClient, build_deck
from mtg_agentic.llm import OllamaClient

with OracleStore() as store, ScryfallClient() as client:
    result = build_deck(
        "Bracket 3 Myojin tribal with proliferate",
        "Jodah, Archmage Eternal",
        OllamaClient(model="qwen2.5:7b"),
        client,
        store,
    )
```

The separate `intake`, `discover`, and `draft_deck` functions accept the same
injectable `JSONModel` protocol, so tests can run entirely without an LLM.

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
Pass `fuzzy=False` for exact-only verification. `get_card_record(name)` retains
rule/mana metadata for local indexing, not LLM context.
`get_rulings(card_id)` retrieves and paginates `/cards/:id/rulings`.
`search_cards` also accepts optional `max_results` and `order="edhrec"` for
bounded discovery; leaving these unset preserves full pagination.

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
Changer list is maintained. Offline validation uses snapshot flags; safety reviews
use a separately dated live Game Changer list.

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
  Grist's special creature status, legendary Vehicles and legendary Spacecraft
  with power/toughness boxes (current CR 903.3). Refresh older snapshots to retain
  the additional power/toughness fields.
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

The offline validator's bracket check covers **Game Changer counts only**, not full bracket
certification: combo speed, extra turns, mass land denial, and playgroup intent
are outside this validator. `valid` means these implemented checks passed
against the recorded snapshot, not that all social bracket expectations were met.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use synthetic records, mocked HTTP, and a fake clock. They require no
network, no bulk download, no Ollama installation, and no third-party test
libraries. Intake, query planning, draft repair, land allocation, provider
transports, dated caches, Spellbook matching, citation rejection, four-case metrics,
and CLI artifacts are covered with scripted model responses.

API references: [Scryfall](https://scryfall.com/docs/api),
[rate limits](https://scryfall.com/docs/api/rate-limits),
[bulk data](https://scryfall.com/docs/api/bulk-data).
Card data belongs to Wizards of the Coast and is provided by Scryfall.

## Bracket review

```powershell
python -m mtg_agentic bracket-check .cache/jodah-deck.json
python -m mtg_agentic review-deck .cache/jodah-deck.json --output .cache/jodah-review.json
```

`bracket-check` combines deterministic signals with one structured LLM judgment
per assessment. `review-deck` also checks legality, ingests every distinct deck
card's rulings and reviews flagged interactions. Verdicts are `fits`, `mismatch`,
or `uncertain`. These are pregame aids, not bracket certification.

Policy was checked against Wizards' [original announcement](https://magic.wizards.com/en/news/announcements/introducing-commander-brackets-beta),
[October 21, 2025 update](https://magic.wizards.com/en/news/announcements/commander-brackets-beta-update-october-21-2025),
and [February 9, 2026 update](https://magic.wizards.com/en/news/announcements/commander-brackets-beta-update-february-9-2026)
on October 25, 2026. Thresholds are versioned in code, not automatically inferred
from future pages; review newer announcements before changing them.

- Game Changers come from live Scryfall `is:gamechanger`, cached for 24 hours
  with an observation date and complete list in each report. The underlying
  Scryfall response may itself be cached for up to 24 hours. This list overrides
  old snapshot flags during reviewed validation.
- Standard Game Changer limits are zero for Brackets 1/2 and three for Bracket 3;
  Brackets 4/5 have no numerical cap. Thematic Bracket 1/Rule Zero exceptions
  require pod agreement and are not auto-approved.
- Mass land denial uses named examples and Oracle patterns, including nonbasic
  land denial and untap locks. Replacement-land effects are review candidates,
  not automatic failures. These heuristics are **not exhaustive**.
- Extra turns and possible recursion are counted. One extra-turn card does not
  prove chaining. Bracket 1 disallows extra-turn cards; likely chains conflict
  with Brackets 2/3.
- Tutors are counted as printed mana value at most two versus more expensive,
  with a separate land-search heuristic. Activation costs and conditional search
  restrictions can differ. **Wizards removed tutor limits in October 2025**;
  neither count has an invented cap. Fast mana is descriptive too.
- The LLM evaluates curve, interaction, likely win speed and combo consistency.
  Wizards' development windows are nine/eight/six/four turns for Brackets 1–4:
  for Bracket 3, a seventh-turn ending is an acceptable expectation. Unknown
  speed or an estimated range overlapping that window yields uncertainty, not
  proof of a hard violation. One unusual game does not determine a bracket.
  Bracket 5's cEDH metagame intent cannot be proven from a decklist.

Known combos are credited to [Commander Spellbook](https://commanderspellbook.com).
The client sends sparse, read-only `/find-my-combos` POST searches with deck names
and quantities, a named User-Agent, at least 800 ms between requests, and a dated
24-hour cache. No account/key is needed. At most three pages are requested per
check; 429 cooldowns are respected without automatic retries.

Only exact included, working, two-distinct-card variants are used. Commander
requirements, quantities, starting states, steps and prerequisites are preserved.
Almost-included combos are not deck combos. Templates and truncated searches make
completeness uncertain. Multi-card, unknown and unrecorded combos remain outside
this check. Inclusion does not prove the prerequisites are achievable or that
an early win is consistent.

## Rules and rulings RAG

```powershell
python -m mtg_agentic download-rules
python -m mtg_agentic download-rules --refresh
python -m mtg_agentic check-interaction "Jodah, Archmage Eternal" "Myojin of Seeing Winds" "Contagion Clasp" --question "Does casting this Myojin from hand using Jodah give it a divinity counter, and can proliferate replace a counter after the last one is removed?"
```

The downloader discovers the current TXT link on the official
[rules page](https://magic.wizards.com/en/rules), records its effective date, and
stores the raw response locally. The September 25, 2026 text was live-tested and
yielded 3,165 numbered chunks. TOC entries and the glossary do not leak into rules.
Chunks are embedded locally with Ollama `nomic-embed-text` and stored in SQLite.
Retrieval combines vector cosine similarity with lexical overlap, reserving
context for both rules and card-specific Oracle text/publisher rulings.

Scryfall rulings are ingested for **every distinct card, including the commander
and basic lands**, even when a card has no published rulings. Unchanged evidence
is not re-embedded. Indexes are tied to the embedding model; use a new database
when changing it. Review commands accept `--embedding-model` and `--embedding-url`.
`--embedding-backend hash` explicitly selects a dependency-free hashed-token
**lexical baseline**, not a semantic model or silent fallback. It uses
`rules-hash.sqlite3`; the default Ollama backend uses `rules.sqlite3`.
Both stay local even when the chat provider is hosted.

`check_interaction` requires per-claim retrieved rule/ruling IDs and verbatim
quotes. Missing IDs, invented quotes, uncited conclusions, and Oracle-only rule
arguments return `unsupported` without asserted claims. Every accepted claim
needs a retrieved CR rule or publisher ruling. Reports retain evidence, source
URLs and dates. Reviews account for hand versus other zones, alternate costs,
replacement effects and existing counter types.

Draft reviews check each included Myojin engine, a proliferate/poison question
when relevant, and detected combos, with a 20-interaction budget. Overflow
requires manual review rather than silently passing. Citation existence and
quotes are checked in code; **logical support remains LLM judgment**, not a
rules-engine proof. Unsupported/contradicted findings enter bounded repairs;
network/model failures never become successful reviews.

Reviewed Python usage:

```python
from mtg_agentic import (
    BracketChecker,
    DeckReviewer,
    OllamaEmbedder,
    OracleStore,
    RulesStore,
    ScryfallClient,
    SourceCache,
    build_deck,
    check_interaction,
)
from mtg_agentic.llm import OllamaClient

model = OllamaClient()
with (
    OracleStore() as store,
    ScryfallClient() as client,
    SourceCache(".cache/mtg-agentic/sources.sqlite3") as sources,
    RulesStore(".cache/mtg-agentic/rules.sqlite3", OllamaEmbedder()) as rules,
):
    rules.download_rules(sources)
    reviewer = DeckReviewer(
        BracketChecker(store, client, sources, model), rules, model, store, client
    )
    result = build_deck(
        "Bracket 3 Myojin tribal",
        "Jodah, Archmage Eternal",
        model,
        client,
        store,
        reviewer=reviewer,
    )
    cards = ["Jodah, Archmage Eternal", "Myojin of Seeing Winds"]
    rules.ingest_deck(cards, store, client)
    interaction = check_interaction(
        cards, "Does the alternate cost preserve casting from hand?", rules=rules, model=model
    )
```

## Four-case evaluation

```powershell
python -m mtg_agentic evaluate --label qwen-local --runs-dir .cache/evaluations
python -m mtg_agentic evaluation-history --runs-dir .cache/evaluations
```

Each evaluation runs intake, discovery, drafting/repairs and full reviews:

| Case | Commander | Bracket | Required theme coverage |
|---|---|---|---|
| Jodah/Myojin | Jodah, Archmage Eternal | 3 | At least 8 distinct Myojin |
| Poison | Atraxa, Praetors' Voice | 3 | At least 8 poison/infect/toxic/proliferate cards, including 3 poison-seeding cards |
| Politics | Queen Marchesa | 3 | At least 8 monarch/goad/vote/offer cards |
| Deliberately lower-power | Ruxa, Patient Professor | 2 | At least 8 vanilla creatures |

Theme coverage is a transparent Oracle-text/name heuristic, not a gameplay quality
score. The poison seed heuristic checks infect/toxic or poison-giving wording;
review unusual conditional effects manually.

Every run writes a new UUID directory with deck artifacts and `report.json`,
plus a record in `evaluations.sqlite3`. History tracks validator pass rate,
bracket accuracy (target-label agreement with a `fits` assessment), capped theme
coverage, theme pass rate, safety-review pass rate and overall case pass rate.
Legality is independently rerun; generated `valid` flags are not trusted.
Failures remain in the four-case denominator, including model/HTTP failures.
Artifacts retain card, policy and rules provenance. Model, embedding and build
settings are recorded automatically; labels can identify additional experiment details.
Python `run_evaluation(runner, store, directory)` accepts a scripted runner for
offline harness testing.

A local stability indicator requires **three consecutive fully passing runs**
with the same model/build configuration and Oracle snapshot.
This is an engineering stability indicator, not Wizards policy. Add
`--forge-home <app-directory> --pod examples/forge-pod.json --games 10` to
`evaluate` to simulate each case after fresh validation, bracket/rules review
and theme checks. Repeat `--pod` for up to three distinct fixed pods.
Failed reviews never reach simulation. Simulation failures stay in the
four-case denominator; reports retain win rate, win-turn distribution,
completion/timeout rates, confidence intervals and available observations per pod.
History includes these metrics, Git commit, dirty-tree digest and Oracle metadata,
so evaluations can be run and compared for each commit. This does not install
a commit hook or automatically launch long simulations.

Evaluation also rejects reported revision gate violations. Regression tests
exercise the invariant that an illegal or unreviewed revision is never simulated
or accepted. Politics primarily uses package/theme and rules checks, not AI
negotiation success.

Live data checks verified 53 Game Changers, a known Spellbook two-card win and
current rules ingestion. Live Ollama generation/embeddings and end-to-end
evaluation quality have **not** been verified in this environment.

## Local Forge installation and smoke test

The tested release is **Forge 2.0.15**, with a separate Java installation.
The local smoke used Oracle Java 24.0.1, 2 GiB JVM heap, and the official archive:

[forge-installer-2.0.15.tar.bz2](https://github.com/Card-Forge/forge/releases/download/forge-2.0.15/forge-installer-2.0.15.tar.bz2)

SHA-256: `4468f182b60f6ca62d89697baa208981085af744f116781af751c611826db600`

Verify the archive hash before extracting it into a new directory. The app
directory must contain `forge-gui-desktop-2.0.15-jar-with-dependencies.jar`
and `res/cardsfolder/cardsfolder.zip`. The current development branch's
headless flags are not interchangeable with this release.

For isolation, create `forge.profile.properties` in that new app directory
**before the first launch**, choosing fresh absolute locations:

```properties
userDir=D:/your-workspace/.cache/forge-2.0.15/profile
cacheDir=D:/your-workspace/.cache/forge-2.0.15/profile/cache
```

Do not overwrite an existing profile or share it with an interactive Forge
session during experiments. Exports default to that profile's
`decks/commander` folder. `--decks-dir` explicitly overrides the destination.
Without an isolated profile, Forge may initialize its normal user directories.

Check the actual installed help, using the app directory as the working
directory. Its Windows launcher does not forward these arguments:

```powershell
java '-Djava.awt.headless=true' -jar forge-gui-desktop-2.0.15-jar-with-dependencies.jar sim -h
```

The smoke used `-f Commander`, four `-d` filenames, `-D` deck directory,
`-n 10`, `-q`, `-c 45`, and `-s 20261025`. All ten printed outcomes were
inspected before implementing parsers. Six hit the internal clock yet
**incorrectly printed winners**, sometimes multiple winners. Those are
timeouts, never wins. Four games completed cleanly, with no candidate wins.
See [the observed fixtures](tests/fixtures_forge/README.md).
A subsequent isolated, verbose game completed cleanly with all four decks
supported locally. These were plumbing tests, **not a reviewed strategy
baseline**: the fixture contained Bracket-3-incompatible Myojin of Infinite Rage.

## Export, support reports and simulations

```powershell
python -m mtg_agentic forge-support deck.json --forge-home .cache/forge-2.0.15/app --output support.json
python -m mtg_agentic forge-export deck.json --forge-home .cache/forge-2.0.15/app --output export.json
python -m mtg_agentic forge-sim deck.json --forge-home .cache/forge-2.0.15/app --pod examples/forge-pod.json --games 10 --timeout 120 --output simulation.json
```

Output files and `.dck` exports are exclusive, never overwritten. Export validates
99+1, legality and color identity, writes a separate `[Commander]`, and uses
front-face Forge names. Partner/background configurations remain unsupported.
Export and raw simulation do **not** certify bracket/rules review; this permits
explicit plumbing tests. Revisions and simulation evaluations require fresh
safety gates regardless of saved `valid`/`review` flags.

`ForgeSupport` checks every distinct card against local archived/loose scripts
and expected faces. Reports list missing scripts, manually recorded known
issues and AI hints. `--known-issues issues.json` accepts an exact-card-name to
observed-issue-description mapping and blocks affected decks. Script presence
is not a rules certificate. `AI:RemoveDeck` hints warn about AI/deck generation;
they are not proven card bugs. Runtime target-selection/default-choice/FightAi
warnings are recorded separately, not silently diagnosed as deck weaknesses.

The example pod pins the release and SHA-256 of three bundled precons:
Built from Scratch, Plunder the Graves and Swell the Host. Changed opponent
files fail before simulation. Comparisons reuse the same pod, seeds, clock,
Forge JAR/scripts, Oracle metadata and seat rotation. Do not change AI settings
or the isolated profile between comparison batches. Seeds do not guarantee
bit-for-bit results from Forge's threaded AI.

Each game gets a fresh JVM and seed, with candidate seats rotating 1, 4, 3, 2.
This costs startup time but prevents an interrupted game from contaminating
later games. The wrapper uses no shell, bounds raw-log size and kills its own
process tree on hard timeout or cancellation. `--timeout` is Forge's game clock;
the outer deadline adds `--startup-grace` (default 30 seconds).
`ForgeSimulator.cancel()` stops an active run. Raw logs, hashes, support reports,
export hashes and manifests persist in unique simulation directories.
Exit 0 requires every requested game to have a clean parsed result, not merely
a successful Java exit.

Tier 1 reports completed-only win rate with a 95% Wilson interval, wins per
requested game, win-turn distribution, completion rate and timeout rate.
Failures are neither wins nor deck losses. A low completion rate can bias
completed-only results. Forge's outcome turn counter and verbose global turn
counter differ; neither is relabeled as multiplayer rounds.

Use `--verbose-logs` to retain turn/cast events. An exact commander first-cast
event can then be observed in a complete turn trace, with its global counter
and the candidate's own turn-start count. This is not proof of resolution or
command-zone provenance. Opening lands, battlefield lands by turn 4,
screw/flood, draw/ramp by turn 5 and a concrete
Myojin-with-divinity-plus-proliferate state by turn 6 remain explicitly
**unavailable**. Card names, replacement-effect descriptions or casting a ramp
spell are not fabricated state measurements.

Python entry points include `export_deck`, `ForgeSupport`, `ForgeInstallation`,
`OpponentPod`, `ForgeSimulator`, `simulate` and `summarize_simulation`:
`simulate(deck, opponents, n_games, simulator=runner)` returns a persisted raw
manifest; `summarize_simulation(manifest)` verifies log hashes before parsing.
The parser is scoped to inspected 2.0.15 output; unknown releases fail closed.

## Bounded simulation-driven revisions

```powershell
python -m mtg_agentic revise deck.json --pool candidates.json --forge-home .cache/forge-2.0.15/app --pod examples/forge-pod.json --games 50 --iterations 3 --plateau 2 --output revision.json
```

This command needs the local chat and embedding models used by normal reviews.
It does not run automatically after `build`. Each iteration:

1. Freshly validate/review the baseline, then run about 50 games per fixed pod.
   Stop before diagnosis when completion is below 95%.
2. Ask the model for the top one or two evidence-linked issues, distinguishing
   simulation faults from deck hypotheses. Allow at most five reasoned,
   one-card swaps from the locally rehydrated candidate pool.
3. Keep commander, bracket, inclusions/exclusions and pool-tagged
   theme/mechanics/win-condition package counts fixed or improved. Re-run
   independent validation and full bracket/rules review on the actual revision.
4. Simulate only passing revisions, with the same pods, seeds and seats.
   Require a positive conservative Wilson difference bound **in every pod**.
   Confidence levels account for the predeclared number of pods and iterations.
   Unknown baseline outcomes favor the baseline; unknown candidate outcomes
   disfavor the candidate.
5. Reject observed completion/timeout or candidate win-turn-median regressions.
   Accept only clear improvements; otherwise keep the previous deck. These
   secondary guards are observed vetoes, not statistical equivalence proofs.

Subsequent iterations use fresh paired seed batches. Two non-improvements stop
the default run; the default maximum is three iterations. A one-iteration run
also needs `--plateau 1`. Every baseline, diagnosis, proposed deck, gate report,
candidate result and decision is persisted, together with the final deck,
commit/dirty-tree identity and model configuration. Gates and audit fields are
tested independently; unsupported interactions cannot be bypassed by a model's
saved review flag.

This is deliberately conservative: 50 games often cannot distinguish small
gains. A default one-pod run can launch up to 300 games, each in its own JVM,
so budget CPU time accordingly. Package checks cannot establish real table
politics or human enjoyment. Live LLM diagnosis and 50-game optimization have
**not** been demonstrated here; acceptance/revert and gate invariants are
covered with explicitly scripted tests.
