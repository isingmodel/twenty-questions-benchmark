# Twenty Questions Benchmark

A polished multi-turn benchmark for measuring how efficiently LLMs can solve a hidden-target game through yes/no questions.

This benchmark is meant to capture something many static QA leaderboards miss: interactive search under uncertainty. To succeed, a model has to ask discriminative yes/no questions, update its internal hypothesis after each answer, and decide when to stop exploring and commit to a direct identity check. In practice, that makes it a compact test of question selection, hypothesis tracking, and budgeted decision-making rather than pure one-shot recall.

At a deeper level, the benchmark is trying to measure a model's ability to reduce uncertainty through dialogue: ask high-information questions, maintain and revise a working hypothesis over multiple turns, and convert that narrowed search space into a precise identification at the right moment. That makes the repository a useful bridge between simple single-turn evals and looser agent benchmarks.

One model acts as the guesser. Another acts as the judge. The protocol is intentionally explicit: target records, prompt templates, judge policy, reasoning settings, and logs are all inspectable, so runs can be compared under a clearly specified setup and audited from full transcripts rather than a black-box win rate alone. Every run produces prompts, event logs, transcripts, suite aggregates, and analysis-ready reports.

## Why This Benchmark Matters

- It evaluates interactive uncertainty reduction instead of single-shot recall.
- It keeps the protocol fixed enough that side-by-side model comparisons are interpretable.
- It logs full trajectories, so you can inspect search behavior and failure modes rather than only win/loss outcomes.
- It is small enough to rerun frequently, but rich enough to expose meaningful differences in questioning strategy.

## How It Works

```text
    Guesser                      Judge
       |                           |
       |--- "Is it a place?" ----->|
       |<-- {"label":"Yes"} -------|
       |                           |
       |--- "Is it in Europe?" --->|
       |<-- {"label":"Yes"} -------|
       |                           |
       |--- "Is it Paris?" ------->|
       |<-- {"label":"Yes"} -------|  => SOLVED in 3 turns
```

- There is no separate "final guess" phase.
- The guesser wins by asking a direct identity-check question that the judge confirms.
- Every turn is logged with prompts, raw outputs, judgments, latency, and transcript artifacts.

## Benchmark Results

The table below is a snapshot of the checked-in `results/results.csv`. Labels follow the run-level identifiers used by the plotting scripts and may reflect either named reasoning efforts or explicit thinking budgets.

| Rank | Model | Solve Rate | Avg Turns / Success | Runs |
|-----:|-------|----------:|--------------------:|-----:|
| 1 | Claude Opus 4.6 (budget 2048) | 99.29% | 21.73 | 140 |
| 2 | GPT-5.4 (low) | 98.57% | 23.10 | 140 |
| 3 | GPT-5.4 Mini (high) | 98.57% | 24.09 | 140 |
| 4 | GPT-5 (low) | 98.56% | 22.20 | 139 |
| 5 | Gemini 3.1 Flash Lite | 93.57% | 25.55 | 140 |
| 6 | GPT-5.4 Mini (low) | 93.57% | 28.24 | 140 |
| 7 | Claude Sonnet 4.5 (budget 2048) | 90.71% | 22.90 | 140 |
| 8 | Gemini 3 Flash | 88.57% | 21.56 | 140 |
| 9 | GPT-4o | 83.57% | 21.87 | 140 |


**Snapshot takeaways:**

- **Claude Opus 4.6** remains the clearest all-around leader, combining a near-perfect solve rate (99.3%) with low turn count (21.7 turns per success).
- **GPT-5 and GPT-5.4 are the strongest OpenAI variants in this snapshot.** GPT-5 (`low`) nearly matches the top solve rate while staying materially more efficient than GPT-5.4 (`low`) on successful games.
- **Reasoning effort matters for GPT-5.4 Mini.** Moving from `low` to `high` improves solve rate by 5 percentage points and trims about 4 successful turns on average.
- **Fast successful solves are not enough on their own.** Gemini 3 Flash and GPT-4o solve quickly when they do succeed, but their lower solve rates keep them well outside the top tier of the overview plot.
- All models were judged by the same judge configuration, so the differences shown here are best read as guesser-side behavior under a fixed protocol rather than judge variance.

The checked-in overview plot below is generated from `results/results.csv`.

![Model Performance Overview](img/model_overview.png)

See [Reproducibility](docs/reproducibility.md) for the broader workflow and reporting expectations.

## DWEI Metric (Difficulty-Weighted Efficiency Index)

### Why a specialized metric?

Solve rate and average turns are useful but incomplete. 
- A model that solves 95% of games in 40 turns each is arguably worse than one that solves 90% in 15 turns. 
- When a model *fails* to solve a target, it hits the maximum budget (e.g., 40 or 80 turns). Excluding these failures causes extreme **survivor bias**, making models that only solve easy problems look artificially fast.
- Easy problems and hard problems take vastly different numbers of turns. A simple turn difference on an easy problem should not carry the same absolute weight as a difference on a hard problem.

DWEI addresses these issues by leveraging **survival analysis** and **difficulty-weighting** to create a mathematically robust and intuitive "efficiency index".

### How it works

1. **Calculate Target RMQ ($R_{m,i}$):** Each game yields `(turns_used, solved)`. For each `target × model` combination, we fit a Kaplan-Meier survival curve up to the target's maximum recorded turn horizon. The integral (area under this curve) is the Restricted Mean Questions (RMQ), representing the expected number of turns to solve. This cleanly penalizes failures without invoking survivor bias.
2. **Determine Problem Difficulty ($D_i$):** For descriptive reporting, a target's difficulty is the arithmetic-mean RMQ across all evaluated models for that target. Higher $D_i$ means the problem was universally harder.
3. **Build a Target Efficiency Baseline ($B_i$):** To normalize scores, we use the **harmonic mean** of per-model RMQs on that target:
   `B_i = HarmonicMean(R_{m,i})`
   This is the right baseline for a ratio-of-speeds metric, and it guarantees that the average model score remains exactly centered at 100.
4. **Calculate Difficulty-Normalized Efficiency ($B_i / R_{m,i}$):** We calculate a speed ratio for each model on each target:
   `Efficiency = Baseline RMQ / Target RMQ`
   If a hard target has a baseline RMQ of 40 and a model solves it in 20 expected turns, its efficiency ratio is 2.0. If an easy target has a baseline RMQ of 10 and the model solves it in 5 expected turns, the ratio is also 2.0.
5. **Final Index (DWEI):** We compute the unweighted mean of this efficiency ratio across all targets, then multiply by 100.
   `DWEI = 100 × Mean( B_i / R_{m,i} )`

### Interpreting the score

A score of **100** represents the exact benchmark-average speed across the evaluated model field. 
A score of **120** means the model is, on average, solving these targets **20% faster / more efficiently** than the benchmark baseline. 

### Snapshot DWEI rankings

| Rank | Model | DWEI Score |
|-----:|-------|-----------:|
| 1 | Claude Opus 4.6 (budget 2048) | 116.5 |
| 2 | GPT-5 (low) | 115.3 |
| 3 | GPT-5.4 (low) | 108.5 |
| 4 | GPT-5.4 Mini (high) | 105.9 |
| 5 | Gemini 3 Flash | 95.6 |
| 6 | Claude Sonnet 4.5 (budget 2048) | 93.7 |
| 7 | GPT-4o | 90.8 |
| 8 | Gemini 3.1 Flash Lite | 89.6 |
| 9 | GPT-5.4 Mini (low) | 84.0 |

Under the harmonic-RMQ normalization, **100** is the exact benchmark-average speed across the evaluated field. That makes the mid-table easier to read: **Gemini 3 Flash** sits just below average because its fast successful runs are offset by a lower solve rate, while **Claude Sonnet 4.5** lands in a similar band for the same basic reason. **GPT-4o** and **Gemini 3.1 Flash Lite** are also close enough in score that the ordering should not be over-interpreted; the more durable conclusion is that both cluster below the benchmark average, while **GPT-5**, **GPT-5.4**, and **Claude Opus** form the clearly above-baseline group.

### Generate the plot

```bash
python3 -m analysis.plot_weighted_efficiency \
  --input results/results.csv \
  --output img/weighted_efficiency_ranking.png
```

![DWEI Model Ranking](img/weighted_efficiency_ranking.png)

## Typical Workflows

- Run a single target game and inspect the full transcript
- Run repeated evaluation suites across multiple models and targets
- Aggregate many suite runs into a single benchmark report
- Regenerate a leaderboard-style overview plot from fresh results

## Targets

21 targets across 6 domains:

| Domain | Targets |
|--------|---------|
| animals | elephant, eagle, octopus, platypus |
| characters | Sherlock Holmes, Gandalf |
| foods | pizza, croissant |
| objects | toothbrush, refrigerator, umbrella, bicycle, laptop, violin, stapler |
| people | Marie Curie, Abraham Lincoln |
| places | Paris, Busan, volcano, Sahara Desert |

Target records live in [`data/all_target.csv`](data/all_target.csv) and are validated against [`schemas/target.schema.json`](schemas/target.schema.json).

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for the providers you want to test

Create a `.env` file:

```bash
gemini_key=...
OPENAI_API_KEY=...      # or openai_key
CLAUDE_API_KEY=...      # or ANTHROPIC_API_KEY / anthropic_key
```

### Run a Single Game

```bash
python3 -m twentyq.run_single_game \
  --target-id place_paris \
  --budget 40 \
  --guesser-model gpt-5.4 \
  --judge-model gemini-3-flash-preview
```

By default this writes a new run directory under `runs/`.

### Run a Repeated Suite

```bash
python3 -m twentyq.run_single_target_suite \
  --config configs/single_target_suites/evaluation_v3.json
```

This writes a timestamped suite directory under `reports/single-target-suite/`.

### Run Cross-Suite Analysis

```bash
python3 -m analysis.analyze_single_target_suite --completed-only
```

This writes:

- `reports/single-target-suite/benchmark-analysis/aggregate.json`
- `reports/single-target-suite/benchmark-analysis/report.md`

### Regenerate the Overview Plot

```bash
python3 -m analysis.plot_model_overview
```

By default this reads `results/results.csv` and writes `img/model_overview.png`.

### Reasoning Configuration

```bash
python3 -m twentyq.run_single_game \
  --target-id object_toothbrush \
  --budget 20 \
  --guesser-model gemini-2.5-flash \
  --guesser-thinking-budget 512 \
  --judge-model gemini-3-flash-preview \
  --judge-thinking-level low
```

## Output & Logging

A single game writes one run directory under `runs/` by default. A repeated suite writes a timestamped directory under `reports/single-target-suite/`, with per-run logs under that suite's `runs/` subdirectory.

Each single-game run directory contains:

| Artifact | Description |
|----------|-------------|
| `run_config.json` | Run configuration |
| `summary.json` | Outcome summary |
| `events.jsonl` | Turn-by-turn event log |
| `episodes/<target>.json` | Full transcript and metadata |
| `episodes/<target>.md` | Human-readable transcript |

Suite runs additionally produce:

| Artifact | Description |
|----------|-------------|
| `manifest.json` | Planned targets, variants, repetitions, and resolved reasoning settings |
| `status.json` | Progress and active-run status |
| `results.json` | Per-run records |
| `aggregate.json` | Per-target and per-variant aggregates |
| `report.md` | Markdown summary for the suite |

Cross-suite analysis writes `aggregate.json` and `report.md` under `reports/single-target-suite/benchmark-analysis/`.

## Interpretation

This repository is best used as a controlled interactive benchmark:

- the prompt scaffold is fixed and intentional
- results depend on the chosen judge model and judge prompt
- the target set is explicit and relatively small
- provider-native multi-turn API behavior is part of what gets measured

That makes the project useful for side-by-side comparisons, regression tracking, and protocol experiments. Results should be read as performance inside this benchmark design, not as a universal ranking of model intelligence.

## Repository Layout

```text
twentyq/
  episode_runner.py               # shared gameplay engine
  run_single_game.py              # single-target CLI
  run_benchmark.py                # one-pass benchmark runner
  run_single_target_suite.py      # repeated suite runner

analysis/
  analyze_single_target_suite.py  # cross-suite aggregation
  plot_model_overview.py          # solve-rate vs turns scatter plot
  plot_weighted_efficiency.py     # DWEI ranking plot

configs/single_target_suites/     # suite configuration files
data/                             # target records
docs/                             # benchmark scope and reproducibility notes
img/                              # generated and checked-in images
prompts/                          # guesser and judge prompt templates
reports/                          # generated run outputs (gitignored)
results/                          # checked-in run-level CSV snapshots
schemas/                          # target schema
tests/                            # unit tests
```

## Documentation

- [Benchmark Design](docs/benchmark-design.md)
- [Prompt Scaffold](docs/prompt-scaffold.md)
- [Reproducibility](docs/reproducibility.md)

## License

MIT
