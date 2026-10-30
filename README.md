# Ring

In Depth docs [here](https://ring-2.gitbook.io/ring-docs)

Ring builds Commander decks for Magic: The Gathering from a few words. Name a commander,
describe the deck you want, answer a few questions about playstyle, and a team of agents
searches the card pool, maps the deck's engine, drafts it, and checks it before you see it.

**Models propose, code decides.** A model reads your words and picks cards; code checks every
answer against the real card data and the rules. A weak model makes a weaker deck, never an
illegal one that passes.

- **Interview and quoted asks.** Ring asks about playstyle, then keeps your exact words in
  front of every agent and checks the finished deck serves each of them.
- **Local card search.** Scryfall's card data, tags and rulings and Commander Spellbook's combos
  (up to three cards) are downloaded once, so a 1,000-card candidate pool builds in seconds.
  Discovery fuses a word search over rules text, a card-graph walk from the commander (shared
  tags, keywords and combos) and, with the card index setup builds, EmbeddingGemma's sense of
  what the deck is about. Deck roles come from Scryfall's human-made tags first.
- **Engine map and council.** A model traces the deck's resource loop; an orchestrator sets the
  plan and the role counts it needs, and three members rank their picks in parallel and debate
  only the cards they disagree on. The council is on by default.
- **Picks code respects.** Drafters rank their cards as core, strong and flex. Code fills the
  deck in that order, swaps only flex picks to reach a role count, and tells a repair every pick
  it left out and why. Lands follow the deck's curve.
- **Drafting with checks.** On Claude, the drafter checks its picks with code (roles, curve,
  lands, problems) and searches the card data before it commits, then the orchestrator polishes
  the finished deck once.
- **Checked results.** Legality, a code-built mana base, a bracket review, and rules answers
  that must quote the Comprehensive Rules they rely on. The engine map's own links are checked
  against the rules before anyone drafts; on Claude the judge looks rules up by number or topic
  itself, and settled answers are remembered across reviews and sessions.
- **Follow-ups.** Ask about the deck or for changes; every swap is checked again and saved as a
  new version.

```mermaid
flowchart LR
    A[Interview] --> B[Intake] --> C[Discovery] --> D[Engine map] --> E[Draft or council]
    E --> F{Validation and review}
    F -->|problems| E
    F --> G[Deck] --> H[Follow-up edits]
```

## Quick start

Requires Python 3.10 or newer.

```sh
git clone https://github.com/ToonLink123/MTG-Agentic-AI.git
cd MTG-Agentic-AI
python -m pip install -e ".[tui,embeddings,claude]"
python -m mtg_agentic tui
```

On its first start, Ring downloads what it reads locally, about 2.3 GB, with a checklist
(run `python -m mtg_agentic setup` to do it ahead of time). Then type a commander's name and
describe your deck.

Claude is the default: put `CLAUDE_API=<your key>` in a `.env` file at the repository root; it
is ignored by Git. To stay free and offline, pass `--provider ollama` and use a local model
through [Ollama](https://ollama.com) (`ollama pull qwen2.5:7b`).

## Models

| Provider | Runs on | Default model | Candidate pool |
| --- | --- | --- | --- |
| `ollama` | Your machine | `qwen2.5:7b` | 350 cards |
| `llama-cpp` | Your machine | LFM3 1.7B | 350 cards |
| `claude` (default) | Anthropic's API | `claude-haiku-5-5`; the council uses Sonnet 5.5 to lead | 1,000 cards |
| `openai-compatible` | Any compatible API | the one you name | 350 cards |

Rules embeddings always run locally with Google's EmbeddingGemma 2.

## Terminal app and command line

The terminal app is a chat with live progress, session tabs (Ctrl+N, F6) and `/help` for its
commands. It shows what each run and session cost in tokens, cache share and estimated dollars,
and a deck's curve and each color's pips against its land sources. `/why CARD` says who picked a
card and why; `/versions`, `/diff` and `/undo` walk a deck's saved versions; `/checkpoint on`
pauses before drafting so you can pin or ban cards and change role targets. Every step it runs is
also a command you can run yourself, each reading the file the last one wrote (`--events` prints
its progress as JSON lines, which is how the terminal app reads it):

```sh
python -m mtg_agentic intake --commander "Jodah, Archmage Eternal" --request "Bracket 3 Myojin tribal" --output spec.json
python -m mtg_agentic discover --spec spec.json --output pool.json
python -m mtg_agentic engine-map --pool pool.json --output mapped.json
python -m mtg_agentic council-plan --pool mapped.json --output plan.json   # optional
python -m mtg_agentic council --pool mapped.json --plan plan.json --output deck.json
```

Results are JSON, files are never overwritten, and `validate` exits with `1` for a deck that
fails its checks. [`examples/`](examples) has a spec, a deck that fails on purpose, and a Forge
pod. Optional [Forge](https://github.com/Card-Forge/forge) simulation, a four-deck evaluation
suite and simulation-driven revisions are also built in.

## Development

```sh
python -m unittest discover -s tests -v
```

The tests need no network, downloads or model. [BENCHMARK.md](BENCHMARK.md) explains how rules
retrieval is measured.

## Credits

Card data and rulings come from [Scryfall](https://scryfall.com); card names and text belong to
Wizards of the Coast. Combos come from [Commander Spellbook](https://commanderspellbook.com),
the [Comprehensive Rules](https://magic.wizards.com/en/rules) from Wizards of the Coast, and
rules embeddings use [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).
