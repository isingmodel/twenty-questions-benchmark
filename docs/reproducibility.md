# Reproducibility

## Goal

The repository should make it easy to regenerate reports from code and logged runs instead of relying on hard-coded headline numbers.

## Generate Fresh Results

Run a suite:

```bash
python3 -m twentyq.run_single_target_suite \
  --config configs/single_target_suites/evaluation_v3.json
```

This creates a timestamped output directory under `reports/single-target-suite/`.

Aggregate completed suites:

```bash
python3 -m twentyq.analyze_single_target_suite --completed-only
```

This writes:

- `reports/single-target-suite/benchmark-analysis/aggregate.json`
- `reports/single-target-suite/benchmark-analysis/report.md`

Regenerate the overview plot:

```bash
python3 -m twentyq.plot_model_overview
```

By default the plot script reads:

- `reports/single-target-suite/benchmark-analysis/aggregate.json`

and writes:

- `img/model_overview.png`

## Reporting Expectations

When reporting benchmark results:

- cite the suite config that produced them
- keep the reported target set, budget, repetitions, and judge model explicit
- prefer generated `report.md` and `aggregate.json` over hand-maintained summary tables
- avoid claiming a result bundle exists unless the corresponding artifact path is actually present

## Consistency Checklist

- README paths should match real generated output paths.
- Plot scripts should consume the current analysis schema.
- Documentation should describe currently available commands and files, not planned ones.
