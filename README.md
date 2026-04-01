# Twenty Questions Benchmark

A multi-turn benchmark that evaluates how efficiently LLMs can identify a hidden target through yes/no questions -- the classic game of Twenty Questions, played between AI models.

One **stateful guesser** asks questions. One **stateless judge** answers `Yes`, `No`, or `Ambiguous`. The game is solved when the guesser correctly identifies the target.

![Model Performance Overview](img/model_overview.png)

## Key Results

309 games across 4 guesser models and 4 targets (as of April 2026):

| Model | Solve Rate | Avg Turns (solved) |
|-------|----------:|---------:|
| GPT-5.4 | **98.8%** | 21.3 |
| Gemini 3.1 Flash Lite | 92.5% | **18.3** |
| Gemini 3 Flash | 91.3% | 19.1 |
| GPT-5.4 Mini | 86.2% | 20.2 |

GPT-5.4 achieves the highest solve rate. Gemini 3.1 Flash Lite solves games in the fewest turns. See the [full analysis report](benchmark_analysis/claude_benchmark_analysis/report.md) for per-target breakdowns, confidence intervals, and domain-level patterns.

## How It Works

```
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

- The guesser is stateful across turns via provider-native server-side context (Gemini `previous_interaction_id`, OpenAI `previous_response_id`).
- The judge is stateless -- it sees only the hidden target record and the current question each turn.
- There is no separate "final guess" phase. The guesser wins by asking a direct identity-check question that the judge confirms.

## Targets

16 targets across 6 domains:

| Domain | Examples |
|--------|----------|
| animals | elephant, eagle, octopus |
| characters | Sherlock Holmes |
| foods | pizza, kimchi |
| objects | umbrella, toothbrush |
| people | Marie Curie |
| places | Paris, Busan, Tokyo, Sahara Desert |

Target records are defined in [`data/targets/all_targets.csv`](data/targets/all_targets.csv) following [`schemas/target.schema.json`](schemas/target.schema.json). Benchmark splits live in [`data/splits/`](data/splits/).

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for the providers you want to test

Create a `.env` file:

```
gemini_key=...
OPENAI_API_KEY=...
```

### Run a Single Game

```bash
python3 -m twentyq.run_single_game \
  --target-id place_paris \
  --budget 40 \
  --guesser-model gpt-5.4 \
  --judge-model gemini-3-flash-preview
```

### Run a Benchmark Split

```bash
python3 -m twentyq.run_benchmark \
  --split test \
  --budget 80 \
  --guesser-model gemini-2.5-flash \
  --judge-model gemini-3-flash-preview
```

### Run a Repeated Suite

Suite configs define multiple models, targets, and repetitions in a single JSON file:

```bash
python3 -m twentyq.run_single_target_suite \
  --config configs/single_target_suites/evaluation_v3.json
```

### Reasoning Configuration

Control thinking budget or reasoning effort per model:

```bash
python3 -m twentyq.run_single_game \
  --target-id object_toothbrush \
  --budget 20 \
  --guesser-model gemini-2.5-flash \
  --guesser-thinking-budget 512 \
  --judge-model gemini-3-flash-preview \
  --judge-thinking-level low
```

### Cross-Suite Analysis

Aggregate results across all suite runs:

```bash
python3 -m twentyq.analyze_single_target_suite
```

## Repository Layout

```
twentyq/
  episode_runner.py          # shared gameplay engine
  run_single_game.py         # single-target CLI
  run_benchmark.py           # split benchmark orchestration
  run_single_target_suite.py # repeated suite runner
  analyze_single_target_suite.py  # cross-suite aggregation
  plot_model_overview.py          # model performance scatter plot

data/
  targets/all_targets.csv    # target records
  splits/dev.txt, test.txt   # benchmark splits

configs/single_target_suites/ # suite configuration files
prompts/                      # guesser & judge prompt templates
schemas/                      # JSON schema for targets
docs/                         # design docs, protocols, scoring
reports/                      # run outputs (gitignored)
benchmark_analysis/           # analysis scripts & reports
```

## Output & Logging

Each run produces:

| Artifact | Description |
|----------|-------------|
| `run_config.json` | Run configuration |
| `summary.json` | Outcome summary (solved, turns, etc.) |
| `events.jsonl` | Turn-by-turn event log |
| `episodes/<target>.json` | Full game transcript |
| `episodes/<target>.md` | Human-readable transcript |

Suite runs additionally produce `manifest.json`, `results.json`, `aggregate.json`, and `report.md`.

## Documentation

- [Benchmark Design](docs/benchmark-design.md) -- game structure, solve condition, turn semantics
- [Judge Protocol](docs/judge-protocol.md) -- how the judge evaluates questions
- [Scoring](docs/scoring.md) -- metrics and aggregation
- [Dataset Schema](docs/dataset-schema.md) -- target record format
- [Logging](docs/logging.md) -- output format details
- [Multi-turn Design](docs/multiturn-design.md) -- stateful guesser architecture
- [Protocol Modes](docs/protocol-modes.md) -- game mode variants

## License

MIT
