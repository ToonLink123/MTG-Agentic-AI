# Rules retrieval benchmark

Rules review asks a chat model to judge card interactions using **only** the
evidence that retrieval hands it: Comprehensive Rules (CR) text, publisher
rulings and Oracle text. Every claim must quote a retrieved CR rule or ruling
verbatim, so a rule that is not retrieved cannot be cited, and the judge has to
answer `unsupported`. Retrieval recall therefore sets an upper bound on how many
interactions a review can confirm.

This file explains how retrieval was measured before and after two changes:

1. **A real embedding model.** Rules are now embedded locally with Google's
   [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) instead of
   the hashed-token baseline (`HashEmbedder`), which only matches shared words.
2. **A CR graph** (`mtg_agentic/rules_graph.py`). It links each rule to the rules
   it cites ("see rule 601.2f"), to the rule that defines each glossary term it
   uses, and to its parent rule and subrules. Retrieval fuses the embedding,
   word-overlap and graph rankings, and the graph replaced a hand-written list of
   regex patterns that forced a few definition rules into results.

## Setup

- **Corpus:** CR effective September 25, 2026: 3,165 numbered rules, 17,203
  graph edges and 726 glossary/heading terms.
- **Questions:** 18 interaction questions, each with gold rules chosen by hand
  from rules knowledge before looking at any method's output (listed below).
- **Evidence:** the top 18 items, the default `retrieve` limit. The benchmark
  passes no cards, so all 18 slots hold rules. In a real review, half of the
  slots are reserved for the cards' Oracle text and rulings, leaving 9 for rules.
- **Metrics:**
  - `hit@k`: share of questions with at least one gold rule in the top k.
  - `recall@k`: share of each question's gold rules found in the top k, averaged.
  - `MRR`: mean of 1 / rank of the first gold rule (0 when none is retrieved).
- **Hardware:** RTX 3080 Ti, EmbeddingGemma 2 in bfloat16 on CUDA.

## Configurations

| | Retrieval | Vectors | Ranking |
|---|---|---|---|
| A | Previous | Hashed tokens | Cosine + token overlap, plus regex-pinned definition rules |
| B | Hashed tokens + graph fusion | Hashed tokens | Current pipeline |
| C | EmbeddingGemma 2 + stem overlap, no graph | EmbeddingGemma 2 | Current pipeline with an empty graph: no edges and no terms |
| D | EmbeddingGemma 2 + graph fusion (default) | EmbeddingGemma 2 | Current pipeline |

The current pipeline:

1. Ranks rules by embedding cosine and by stem overlap weighted by inverse document
   frequency (IDF), so rare words such as "proliferate" count more than "spell".
2. Seeds a personalized PageRank walk over the graph from three sources: the best
   rules from step 1, the rules that define the question's terms, and the rules
   that define keywords in the cards' Oracle text.
3. Combines the three rankings by reciprocal rank fusion.
4. Keeps the defining rule of each specific term the question names, such as
   "proliferate" or "alternative cost". Terms used by more than 4% of rules are
   ignored.
5. Ranks a multi-part question once in full and once per sentence. The rankings
   then take turns filling the remaining slots.

## Results

| Retrieval | hit@5 | recall@5 | hit@10 | recall@10 | hit@18 | recall@18 | MRR | s/query |
|---|---|---|---|---|---|---|---|---|
| A. Previous: hashed tokens + definition patterns | 0.67 | 0.54 | 0.72 | 0.57 | 0.83 | 0.71 | 0.45 | 0.20 |
| B. Hashed tokens + graph fusion | 0.67 | 0.56 | 0.83 | 0.73 | 0.89 | 0.83 | 0.56 | 0.15 |
| C. EmbeddingGemma 2 + stem overlap, no graph | 0.89 | 0.74 | 1.00 | 0.87 | 1.00 | 0.87 | 0.57 | 0.21 |
| D. EmbeddingGemma 2 + graph fusion (default) | 0.83 | 0.71 | 1.00 | 0.87 | 1.00 | 0.94 | 0.61 | 0.28 |

Per question: rank of the first gold rule / share of gold rules in the top 18
(`-` means no gold rule was retrieved).

| # | Question | Gold rules | A | B | C | D |
|---|---|---|---|---|---|---|
| 1 | Myojin cast from hand with Jodah's alternative cost: still enters with a divinity counter? | 118.9, 118.9a, 118.9c, 614.1c, 614.12 | 2 / 0.40 | 11 / 0.40 | 3 / 0.60 | 4 / 0.60 |
| 2 | Proliferate after the last divinity counter was removed? | 701.34a | 3 / 1.00 | 1 / 1.00 | 5 / 1.00 | 1 / 1.00 |
| 3 | Does a copy also copy +1/+1 counters? | 707.2 | 13 / 1.00 | 10 / 1.00 | 2 / 1.00 | 2 / 1.00 |
| 4 | Deathtouch + trample: is 1 damage per blocker enough? | 702.2c, 702.19b, 702.19c | 1 / 0.67 | 1 / 1.00 | 1 / 1.00 | 1 / 1.00 |
| 5 | Commander would go to the graveyard: command zone instead? | 903.9, 903.9a | 9 / 0.50 | 7 / 1.00 | 3 / 0.50 | 5 / 1.00 |
| 6 | Commander tax on the third cast | 903.8 | 3 / 1.00 | 1 / 1.00 | 1 / 1.00 | 1 / 1.00 |
| 7 | Can I target my own hexproof creature? | 702.11a, 702.11b | - / 0.00 | 2 / 0.50 | 2 / 0.50 | 2 / 1.00 |
| 8 | When does a trigger from mid-cast go on the stack? | 603.2, 603.3, 603.3b | - / 0.00 | - / 0.00 | 8 / 0.33 | 8 / 0.67 |
| 9 | 0 toughness vs indestructible | 704.5f | 15 / 1.00 | 10 / 1.00 | 6 / 1.00 | 8 / 1.00 |
| 10 | Layer of P/T-setting effects vs +1/+1 counters | 613.4, 613.4b, 613.4c | 3 / 0.67 | 1 / 1.00 | 1 / 0.67 | 1 / 0.67 |
| 11 | Mana value of an X spell on the stack | 202.3e | 1 / 1.00 | 1 / 1.00 | 3 / 1.00 | 1 / 1.00 |
| 12 | Token copy of my legendary creature: legend rule? | 704.5j | - / 0.00 | - / 0.00 | 5 / 1.00 | 6 / 1.00 |
| 13 | Damage to a planeswalker and its loyalty counters | 120.3c | 3 / 1.00 | 3 / 1.00 | 2 / 1.00 | 3 / 1.00 |
| 14 | Who orders two replacement effects? | 616.1 | 1 / 1.00 | 3 / 1.00 | 1 / 1.00 | 3 / 1.00 |
| 15 | Haste and a creature I just gained control of | 302.6, 702.10b | 1 / 0.50 | 1 / 1.00 | 1 / 1.00 | 1 / 1.00 |
| 16 | Mana abilities under split second | 702.61a, 702.61b | 1 / 1.00 | 1 / 1.00 | 1 / 1.00 | 1 / 1.00 |
| 17 | Is a divinity counter a keyword counter? | 122.1b | 2 / 1.00 | 1 / 1.00 | 2 / 1.00 | 1 / 1.00 |
| 18 | Partner commanders' colors and color identity | 702.124c, 903.4 | 2 / 1.00 | 2 / 1.00 | 2 / 1.00 | 2 / 1.00 |

## Reading the results

- **The embedding model is the largest gain (A → C).** The hashed baseline found
  nothing for questions 7, 8 and 12, which are worded differently from their
  rules ("target my own hexproof creature", "legend rule"). EmbeddingGemma 2
  found a gold rule for all 18 questions.
- **The graph adds the second and third rule an answer needs (C → D).** Gold
  recall in the top 18 rises from 0.87 to 0.94. Questions 5, 7 and 8 gain their
  missing gold rules through references and subrule links (903.9 from 903.9a,
  702.11a next to 702.11b, 603.2 from 603.3). MRR rises from 0.57 to 0.61 because
  questions 2, 11 and 17 move their defining rule to rank 1.
- **The graph costs a little at the very top.** `hit@5` drops from 0.89 to 0.83:
  question 12's only gold rule moves from rank 5 to rank 6. Questions 1, 9, 13
  and 14 also slip one or two places. A review sends 9 rule slots, so recall at
  that depth matters more than the top 5.
- **The graph also works without a semantic model (A → B).** On the same hashed
  vectors, graph fusion beats the hand-written patterns it replaced: recall in
  the top 18 rises from 0.71 to 0.83 and MRR from 0.45 to 0.56.
- **Cost:** about 0.3 s per question, mostly the Python cosine over 3,165 vectors.
  Embedding the whole CR takes about 20 seconds, including model load.

## Known gap: the long Myojin review question

`DeckReviewer` asks one long, multi-part question per Myojin: when it enters
with its counter, whether Jodah's alternate cost preserves that, and what
proliferate does. For this question, the old regex patterns forced the
alternative-cost and counter definitions in. The new retrieval finds proliferate
and the replacement-effect rules, but not 118.9 or 122.1. The glossary defines
"alternative cost" by pointing at all of section 118, and the best-ranked rule
in that section for this question is 118.12, not 118.9.

| | CR evidence for Myojin of Seeing Winds (9 rule slots) |
|---|---|
| Previous | 118.6a, 118.9, 118.9c, 118.9d, 122.1, 122.1b, 601.2, 614.12, 701.34a |
| Current (LFM3 run r5) | 701.34a, 108.1, 118.12, 614.13b, 616.1f, 601.2b, 702.155a, 608.2d, 614.12 |

The shorter benchmark question 1 still reaches 118.9, 118.9c and 614.12. A
candidate fix is anchoring a term to the rule whose text defines it ("An
alternative cost is…") instead of to the whole section the glossary names.

## End to end: LFM3 1.7B drafts

These are single runs, not a benchmark, and they change several things at once:
retrieval, the step-by-step prompts in `mtg_agentic/prompts.py`, a per-deck bracket
schema, and keeping mass land denial out of Bracket 1-3 pools. They use
`LiquidAI/LFM3-1.7B-GGUF:Q8_0` on llama.cpp, the same Jodah Myojin spec, and up to
4 draft attempts. Each attempt takes about 16 minutes.

| Run | Retrieval | Malformed replies | Deck legal and playable | Bracket check | Interactions supported |
|---|---|---|---|---|---|
| r2 | Hashed tokens + patterns | 0, but 2 bracket answers failed grounding | yes | mismatch, then failed grounding | 0/10 |
| r4 | EmbeddingGemma 2 + graph, new prompts | 1 draft, 2 bracket answers failed grounding | from attempt 2 | mismatch (Myojin of Infinite Rage destroys all lands) | 3/10 |
| r5 | Same as r4, plus per-deck bracket schema and a rebuilt pool | 0 | every attempt | uncertain | 5/9 |

No run produced a deck that passed review. In r5, LFM3 judged the bracket fit
`uncertain` and estimated a turn-6 win, which counts as too early for Bracket 3.
Four interactions still failed because the model misquoted evidence or argued
from Oracle text without citing a rule or ruling.

## Limitations

- 18 questions is small: one question moves `hit@k` by 0.056.
- The gold rules were picked by one person; other rules can also support an answer.
- One CR version, and rules-only queries with no card evidence.
- Timings are from one GPU machine and include query embedding.

## Reproduce

```powershell
python -m pip install -e ".[embeddings]"
python -m mtg_agentic download-rules
python -m mtg_agentic download-rules --embedding-backend hash
python -m benchmarks.rules_retrieval
```

`benchmarks/rules_retrieval.py` copies both indexes to a temporary directory, so
the originals are never modified. It prints both tables above. The previous
algorithm (A) is reimplemented in the script, since it no longer exists in
`mtg_agentic/rules.py`.
