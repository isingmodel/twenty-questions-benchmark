# Benchmark Design

## Scope

This repository implements a narrow interactive benchmark:

- a hidden target is sampled from a small explicit dataset
- a guesser model asks exactly one yes/no-style question per turn
- a judge model answers `Yes`, `No`, or `Ambiguous`
- the episode ends when the guesser makes a direct identity-check question that the judge confirms, or when the budget is exhausted

This is useful for comparing behavior inside a fixed protocol. It should not be presented as a general ranking of LLM intelligence or a provider-independent reasoning leaderboard.

## What Is Fixed

- target records from [`data/all_target.csv`](../data/all_target.csv)
- prompt templates in [`prompts/`](../prompts/)
- run logging and aggregation format

## What Can Vary

- guesser model
- judge model
- target subset
- repetition count
- turn budget
- model-specific reasoning settings

## Important Caveats

- Results are judge-conditioned.
- Results are prompt-conditioned.
- Results depend on each provider's multi-turn API behavior.
- The current target set is small and public.

These are limitations on interpretation, not reasons the benchmark is unusable.
