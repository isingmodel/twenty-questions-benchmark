# Benchmark Design

## What It Measures

This benchmark is designed to probe interactive uncertainty reduction. A strong model is not just retrieving facts; it is choosing high-information questions, updating a working hypothesis after each answer, and switching from exploration to identification at the right moment.

That makes the benchmark closer to a compact search task than a standard single-turn QA task. It is useful when you want to compare how models narrow a hypothesis space through dialogue under a fixed turn budget.

## Scope

This repository implements a narrow interactive benchmark:

- a hidden target is sampled from a small explicit dataset
- a guesser model asks exactly one yes/no-style question per turn
- the guesser runs in a provider-native multi-turn session
- the judge is called statelessly each turn with the full target record and the current question
- the judge answers `Yes`, `No`, or `Ambiguous`
- the episode ends when the guesser asks a direct identity-check question naming the target or an alias and the judge confirms it, or when the budget is exhausted

This is useful for comparing behavior inside a fixed protocol. It should not be presented as a general ranking of LLM intelligence or a provider-independent reasoning leaderboard.

## What Is Fixed

- target records from [`data/all_target.csv`](../data/all_target.csv)
- prompt templates in [`prompts/`](../prompts/)
- the no-separate-final-guess game rule
- run logging and aggregation formats

## What Can Vary

- guesser model
- judge model
- target subset
- repetition count
- turn budget
- model-specific reasoning settings

Reasoning settings are intentionally abstracted at the suite/config level. Depending on the model family, the same nominal effort may resolve to provider-specific controls such as OpenAI `reasoning_effort`, Gemini `thinking_level`, or Anthropic and Gemini `thinking_budget`.

## Execution Modes

The repository currently supports three closely related execution patterns:

- `python3 -m twentyq.run_single_game` for one fully logged episode under `runs/`
- `python3 -m twentyq.run_single_target_suite` for repeated target-by-target comparisons under `reports/single-target-suite/`
- `python3 -m twentyq.run_benchmark` for a single-pass sweep over the full target file under `reports/<provider>-benchmark/`

## Important Caveats

- Results are judge-conditioned.
- Results are prompt-conditioned.
- Results depend on each provider's multi-turn API behavior.
- The current target set is small and public.
- Cross-provider comparisons include differences in session semantics, not just raw answer quality.

These are limitations on interpretation, not reasons the benchmark is unusable.
