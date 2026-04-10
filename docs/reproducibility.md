# Reproducibility

## Goal

The repository should make it easy to regenerate reports from code and logged runs instead of relying on hard-coded headline numbers.

The current codebase has two distinct result layers:

- suite-native JSON and Markdown outputs under `reports/single-target-suite/`
- checked-in run-level CSV snapshots under `results/`, which are used by the README plotting scripts

Those are related, but they are not produced by the same command path. Keeping that distinction explicit avoids documentation drift.

## Generate Fresh Suite Results

Run a suite:

```bash
python3 -m twentyq.run_single_target_suite \
  --config configs/single_target_suites/evaluation_v3.json
```

By default this creates a timestamped output directory under `reports/single-target-suite/`.

To resume an interrupted suite in an existing directory:

```bash
python3 -m twentyq.run_single_target_suite \
  --config configs/single_target_suites/evaluation_v3.json \
  --suite-dir reports/single-target-suite/<existing-suite-dir> \
  --resume
```

Each suite directory contains `manifest.json`, `status.json`, `results.json`, and a `runs/` directory with per-run artifacts.

## Aggregate Suite Results

Aggregate completed suites:

```bash
python3 -m analysis.analyze_single_target_suite --completed-only
```

This writes:

- `reports/single-target-suite/benchmark-analysis/aggregate.json`
- `reports/single-target-suite/benchmark-analysis/report.md`

You can also point the analyzer at specific suite directories with repeated `--suite-dir` arguments, or at specific `results.json` files with repeated `--results-json` arguments.

## Regenerate Plot Artifacts From The Checked-In CSV Snapshot

Regenerate the overview plot:

```bash
python3 -m analysis.plot_model_overview
```

By default the plot script reads:

- `results/results.csv`

and writes:

- `img/model_overview.png`

Regenerate the DWEI ranking plot:

```bash
python3 -m analysis.plot_weighted_efficiency
```

By default this reads:

- `results/results.csv`

and writes:

- `img/weighted_efficiency_ranking.png`

## Export A Run-Level CSV From `reports/all_sessions/`

The repository also includes a helper for flattening logged run folders under `reports/all_sessions/` into a CSV:

```bash
python3 -m analysis.generate_all_sessions_results_csv
```

By default this reads:

- `reports/all_sessions/`

and writes:

- `results/results_all_sessions.csv`

This is separate from the checked-in `results/results.csv` file currently used by the plot scripts.

## Reporting Expectations

When reporting benchmark results:

- cite the suite config that produced them
- keep the reported target set, budget, repetitions, judge model, guesser prompt set, and reasoning settings explicit
- say whether the claim comes from a suite aggregate, a checked-in CSV snapshot, or another derived artifact
- prefer generated `report.md` and `aggregate.json` over hand-maintained summary tables when discussing suite outputs
- avoid claiming a result bundle exists unless the corresponding artifact path is actually present

## Consistency Checklist

- README paths should match real generated output paths.
- Plot scripts should document their real default inputs.
- Suite-analysis docs should distinguish `reports/single-target-suite/` outputs from checked-in `results/*.csv` snapshots.
- Documentation should describe currently available commands and files, not planned ones.
