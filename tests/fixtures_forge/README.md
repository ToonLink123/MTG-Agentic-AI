# Forge 2.0.15 observed output

These excerpts come from a manual four-player run, seed 20261025, ten games,
45-second game clock, quiet output. They are not synthetic engine telemetry.
The timeout excerpt omits most of the stack trace.

| Game | Reported winner | Usable result |
| --- | --- | --- |
| 1 | Built from Scratch | Yes, Forge turn 33 |
| 2 | Agentic Jodah Smoke | No, clock limit and inconsistent winners |
| 3 | Swell the Host | Yes, Forge turn 20 |
| 4 | Swell the Host | Yes, Forge turn 21 |
| 5 | Plunder the Graves | No, clock limit |
| 6 | Swell the Host | Yes, Forge turn 21 |
| 7 | Plunder the Graves | No, clock limit |
| 8 | Plunder the Graves | No, clock limit |
| 9 | Plunder the Graves | No, clock limit |
| 10 | Plunder the Graves | No, clock limit |

The batch process ended without an available exit code. All ten printed results
were inspected before implementing the parser. Clock limits override winner
lines. There is no trustworthy candidate win in this run.

The candidate was a plumbing fixture, not a reviewed strategic baseline:
Myojin of Infinite Rage violates its requested Bracket 3 restrictions.
Do not interpret these results as evidence for or against a validated deck.
The production runner uses one JVM per game to avoid cross-game contamination
after interrupted AI calculations. Turn values are the printed Forge counters,
not an inferred count of multiplayer rounds.

`first-cast.log` retains turn-start and relevant cast events from a subsequent
isolated verbose game, same initial seed, 120-second clock. It completed cleanly
in 114942 ms; Opponent 2 (Plunder the Graves) won. The first Jodah cast occurred
at global turn 20, the candidate's fifth turn start. Its result counter was 37,
whereas the detailed global counter reached 73. These are distinct Forge fields,
not interchangeable measurements. A replacement-effect description mentioning
divinity is not sufficient evidence of current counters and proliferate targets.
