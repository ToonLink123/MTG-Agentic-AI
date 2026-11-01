"""Goldfish engine benchmark: games per second in goldfish-rs/ as threads are added.

For each deck, the benchmark compiles it once, checks that every thread count returns the
same counters for the same seed, then times the engine at each thread count. Times include
starting the process and passing the deck as JSON. Needs the Oracle data and the engine, both
from setup:

    python -m mtg_agentic setup
    python -m benchmarks.goldfish_speed .cache/jodah-claude3.json other-deck.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from mtg_agentic.goldfish import compile_deck, play_games_rust
from mtg_agentic.oracle import DEFAULT_DATA_DIR, OracleStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("decks", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--games", type=int, default=1_000_000, help="Games per timed run")
    parser.add_argument("--check-games", type=int, default=20_000,
                        help="Games for the same-counters check across thread counts")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cores = os.cpu_count() or 1
    threads = sorted({1, 2, 4, 8, 16, cores} - {t for t in (2, 4, 8, 16) if t > cores})
    rows = []
    with OracleStore(args.data_dir / "oracle.sqlite3") as store:
        for path in args.decks:
            deck = json.loads(path.read_text(encoding="utf-8-sig"))
            start = time.perf_counter()
            compiled = compile_deck(deck, store)
            compile_seconds = time.perf_counter() - start
            reference = play_games_rust(compiled, args.check_games, args.turns, args.seed,
                                        threads=1)
            identical = all(
                play_games_rust(compiled, args.check_games, args.turns, args.seed, threads=t)
                == reference for t in threads
            )
            rates = {}
            for t in threads:
                start = time.perf_counter()
                play_games_rust(compiled, args.games, args.turns, args.seed, threads=t)
                rates[t] = args.games / (time.perf_counter() - start)
            start = time.perf_counter()
            play_games_rust(compiled, 1, args.turns, args.seed, threads=1)
            overhead = time.perf_counter() - start
            rows.append({
                "deck": deck["commander"], "identical": identical,
                "compile_seconds": compile_seconds, "process_overhead_seconds": overhead,
                "games_per_second": {str(t): rate for t, rate in rates.items()},
            })
            print(f"{deck['commander'][:34]:34} identical={identical}  "
                  + "  ".join(f"{t} threads={rate:,.0f}/s" for t, rate in rates.items()))
    report = {
        "machine": {"cpu_count": cores, "platform": platform.platform(),
                    "python": platform.python_version()},
        "games": args.games, "turns": args.turns, "seed": args.seed, "decks": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
