from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .data import load_targets
from .prompts import ROOT


DEFAULT_REPORTS_ROOT = ROOT / "reports" / "single-target-suite"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORTS_ROOT / "benchmark-analysis"


@dataclass(frozen=True)
class SuiteInput:
    suite_dir: Path
    manifest: dict[str, Any]
    results: list[dict[str, Any]]
    status: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0


def _counter_rows(counter: Counter[Any]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows: list[dict[str, Any]] = []
    for value, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        rows.append(
            {
                "value": value,
                "count": count,
                "share": count / total if total else 0.0,
            }
        )
    return rows


def _mix_text(counter_rows: list[dict[str, Any]]) -> str:
    if not counter_rows:
        return "-"
    return ", ".join(f"{row['value']} ({row['count']})" for row in counter_rows)


def _planned_runs_for_suite(manifest: dict[str, Any]) -> int:
    targets = manifest.get("targets", [])
    variants = manifest.get("variants", [])
    return sum(_safe_int(variant.get("repetitions")) * len(targets) for variant in variants)


def _summarize_runs(
    rows: list[dict[str, Any]],
    *,
    expected_runs: int,
) -> dict[str, Any]:
    solved_rows = [row for row in rows if row.get("solved")]
    error_rows = [row for row in rows if row.get("error")]
    turns = [_safe_int(row.get("turns_used")) for row in rows]
    solved_turns = [_safe_int(row.get("turns_used")) for row in solved_rows]
    exhausted_rows = [
        row
        for row in rows
        if _safe_int(row.get("turns_used")) >= _safe_int(row.get("suite_budget"))
    ]
    incorrect_final_rows = [row for row in rows if row.get("final_question_correct") is False]
    budgets = Counter(str(row.get("suite_budget")) for row in rows)
    judge_models = Counter(str(row.get("judge_model")) for row in rows)
    suite_names = Counter(str(row.get("suite_name")) for row in rows)
    suite_statuses = Counter(str(row.get("suite_status")) for row in rows)
    variant_labels = Counter(str(row.get("variant_label")) for row in rows)

    return {
        "observed_runs": len(rows),
        "expected_runs": expected_runs,
        "coverage_rate": len(rows) / expected_runs if expected_runs else 0.0,
        "solved_runs": len(solved_rows),
        "unsolved_runs": len(rows) - len(solved_rows),
        "error_runs": len(error_rows),
        "solve_rate": len(solved_rows) / len(rows) if rows else 0.0,
        "error_rate": len(error_rows) / len(rows) if rows else 0.0,
        "avg_turns_used": sum(turns) / len(turns) if turns else 0.0,
        "median_turns_used": median(turns) if turns else 0.0,
        "avg_turns_solved": sum(solved_turns) / len(solved_turns) if solved_turns else None,
        "min_turns_used": min(turns) if turns else 0,
        "max_turns_used": max(turns) if turns else 0,
        "budget_exhausted_runs": len(exhausted_rows),
        "budget_exhaustion_rate": len(exhausted_rows) / len(rows) if rows else 0.0,
        "incorrect_final_question_runs": len(incorrect_final_rows),
        "incorrect_final_question_rate": len(incorrect_final_rows) / len(rows) if rows else 0.0,
        "budget_mix": _counter_rows(budgets),
        "judge_model_mix": _counter_rows(judge_models),
        "suite_name_mix": _counter_rows(suite_names),
        "suite_status_mix": _counter_rows(suite_statuses),
        "variant_label_mix": _counter_rows(variant_labels),
    }


def _suite_inputs(reports_root: Path, *, include_running: bool) -> list[SuiteInput]:
    suites: list[SuiteInput] = []
    for suite_dir in sorted(path for path in reports_root.iterdir() if path.is_dir()):
        manifest_path = suite_dir / "manifest.json"
        results_path = suite_dir / "results.json"
        if not manifest_path.exists() or not results_path.exists():
            continue
        manifest = _read_json(manifest_path)
        results_payload = _read_json(results_path)
        status_path = suite_dir / "status.json"
        status = _read_json(status_path) if status_path.exists() else {}
        suite_status = str(status.get("status", "unknown"))
        if suite_status != "completed" and not include_running:
            continue
        raw_results = results_payload.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError(f"Expected list field 'results' in {results_path}")
        suites.append(
            SuiteInput(
                suite_dir=suite_dir,
                manifest=manifest,
                results=[row for row in raw_results if isinstance(row, dict)],
                status=status,
            )
        )
    return suites


def build_cross_suite_analysis(
    reports_root: Path,
    *,
    targets_dir: Path,
    include_running: bool = True,
) -> dict[str, Any]:
    targets = load_targets(targets_dir)
    suite_inputs = _suite_inputs(reports_root, include_running=include_running)

    all_rows: list[dict[str, Any]] = []
    expected_by_model: defaultdict[str, int] = defaultdict(int)
    expected_by_target: defaultdict[str, int] = defaultdict(int)
    expected_by_model_target: defaultdict[tuple[str, str], int] = defaultdict(int)

    suites: list[dict[str, Any]] = []
    for suite_input in suite_inputs:
        manifest = suite_input.manifest
        suite_status = str(suite_input.status.get("status", "unknown"))
        suite_name = str(manifest.get("suite_name", suite_input.suite_dir.name))
        planned_runs = _planned_runs_for_suite(manifest)
        observed_runs = len(suite_input.results)
        variants = manifest.get("variants", [])
        target_ids = [str(target_id) for target_id in manifest.get("targets", [])]

        for variant in variants:
            guesser_model = str(variant.get("guesser_model"))
            repetitions = _safe_int(variant.get("repetitions"))
            for target_id in target_ids:
                expected_by_model[guesser_model] += repetitions
                expected_by_target[target_id] += repetitions
                expected_by_model_target[(guesser_model, target_id)] += repetitions

        suites.append(
            {
                "suite_dir": str(suite_input.suite_dir),
                "suite_id": suite_input.suite_dir.name,
                "suite_name": suite_name,
                "status": suite_status,
                "budget": _safe_int(manifest.get("budget")),
                "targets": target_ids,
                "variants": [
                    {
                        "label": str(variant.get("label")),
                        "guesser_model": str(variant.get("guesser_model")),
                        "judge_model": str(variant.get("judge_model")),
                        "repetitions": _safe_int(variant.get("repetitions")),
                    }
                    for variant in variants
                ],
                "expected_runs": planned_runs,
                "observed_runs": observed_runs,
                "coverage_rate": observed_runs / planned_runs if planned_runs else 0.0,
                "completed_at": suite_input.status.get("completed_at"),
            }
        )

        for row in suite_input.results:
            target_id = str(row.get("target_id"))
            target = targets.get(target_id, {})
            enriched = dict(row)
            enriched.update(
                {
                    "suite_id": suite_input.suite_dir.name,
                    "suite_name": suite_name,
                    "suite_status": suite_status,
                    "suite_budget": _safe_int(manifest.get("budget")),
                    "target_domain": target.get("domain"),
                    "target_name_resolved": target.get("name", row.get("target_name")),
                }
            )
            all_rows.append(enriched)

    model_summary: list[dict[str, Any]] = []
    for guesser_model in sorted(expected_by_model):
        rows = [row for row in all_rows if row.get("guesser_model") == guesser_model]
        summary = _summarize_runs(rows, expected_runs=expected_by_model[guesser_model])
        summary.update(
            {
                "guesser_model": guesser_model,
                "targets_covered": sorted({str(row.get("target_id")) for row in rows}),
            }
        )
        model_summary.append(summary)
    model_summary.sort(key=lambda row: (-row["solve_rate"], row["avg_turns_used"], row["guesser_model"]))

    target_summary: list[dict[str, Any]] = []
    for target_id in sorted(expected_by_target):
        rows = [row for row in all_rows if row.get("target_id") == target_id]
        summary = _summarize_runs(rows, expected_runs=expected_by_target[target_id])
        target_summary.append(
            {
                "target_id": target_id,
                "target_name": targets.get(target_id, {}).get("name", target_id),
                "target_domain": targets.get(target_id, {}).get("domain"),
                **summary,
                "best_model": None,
            }
        )

    model_target_summary: list[dict[str, Any]] = []
    for guesser_model, target_id in sorted(expected_by_model_target):
        rows = [
            row
            for row in all_rows
            if row.get("guesser_model") == guesser_model and row.get("target_id") == target_id
        ]
        summary = _summarize_runs(rows, expected_runs=expected_by_model_target[(guesser_model, target_id)])
        summary.update(
            {
                "guesser_model": guesser_model,
                "target_id": target_id,
                "target_name": targets.get(target_id, {}).get("name", target_id),
                "target_domain": targets.get(target_id, {}).get("domain"),
            }
        )
        model_target_summary.append(summary)

    model_target_summary.sort(
        key=lambda row: (
            row["target_id"],
            -row["solve_rate"],
            row["avg_turns_used"],
            row["guesser_model"],
        )
    )

    best_by_target: dict[str, str] = {}
    for target_id in sorted(expected_by_target):
        candidates = [row for row in model_target_summary if row["target_id"] == target_id]
        if not candidates:
            continue
        best_row = sorted(
            candidates,
            key=lambda row: (
                -row["solve_rate"],
                row["avg_turns_solved"] if row["avg_turns_solved"] is not None else float("inf"),
                -row["coverage_rate"],
                row["avg_turns_used"],
                row["guesser_model"],
            ),
        )[0]
        best_by_target[target_id] = best_row["guesser_model"]

    for target_row in target_summary:
        target_row["best_model"] = best_by_target.get(target_row["target_id"])

    target_summary.sort(
        key=lambda row: (
            row["solve_rate"],
            -row["avg_turns_used"],
            row["target_id"],
        )
    )

    observed_runs = len(all_rows)
    expected_runs = sum(expected_by_model.values())
    completed_suites = [suite for suite in suites if suite["status"] == "completed"]
    running_suites = [suite for suite in suites if suite["status"] != "completed"]
    budgets = Counter(str(suite["budget"]) for suite in suites)
    unique_judges = Counter(str(row.get("judge_model")) for row in all_rows)

    key_findings: list[str] = []
    if model_summary:
        best_model = model_summary[0]
        key_findings.append(
            "`{model}` has the highest observed solve rate at {solve_rate:.2%} over {observed}/{expected} runs.".format(
                model=best_model["guesser_model"],
                solve_rate=best_model["solve_rate"],
                observed=best_model["observed_runs"],
                expected=best_model["expected_runs"],
            )
        )
        fastest_model = sorted(
            [row for row in model_summary if row["avg_turns_solved"] is not None],
            key=lambda row: (row["avg_turns_solved"], -row["solve_rate"], row["guesser_model"]),
        )[0]
        key_findings.append(
            "`{model}` is the most turn-efficient on solved runs at {turns:.2f} turns on average, but its solve rate is {solve_rate:.2%}.".format(
                model=fastest_model["guesser_model"],
                turns=fastest_model["avg_turns_solved"],
                solve_rate=fastest_model["solve_rate"],
            )
        )
    if target_summary:
        hardest_target = target_summary[0]
        easiest_target = sorted(
            target_summary,
            key=lambda row: (-row["solve_rate"], row["avg_turns_used"], row["target_id"]),
        )[0]
        key_findings.append(
            "`{target}` is the hardest observed target at {solve_rate:.2%} solve rate and {turns:.2f} average turns.".format(
                target=hardest_target["target_id"],
                solve_rate=hardest_target["solve_rate"],
                turns=hardest_target["avg_turns_used"],
            )
        )
        key_findings.append(
            "`{target}` is the easiest observed target at {solve_rate:.2%} solve rate across {runs} runs.".format(
                target=easiest_target["target_id"],
                solve_rate=easiest_target["solve_rate"],
                runs=easiest_target["observed_runs"],
            )
        )
    undercovered_pairs = [
        row for row in model_target_summary if row["coverage_rate"] < 1.0
    ]
    if undercovered_pairs:
        most_incomplete = sorted(
            undercovered_pairs,
            key=lambda row: (row["coverage_rate"], row["target_id"], row["guesser_model"]),
        )[0]
        key_findings.append(
            "`{model}` on `{target}` is still provisional with {observed}/{expected} runs recorded.".format(
                model=most_incomplete["guesser_model"],
                target=most_incomplete["target_id"],
                observed=most_incomplete["observed_runs"],
                expected=most_incomplete["expected_runs"],
            )
        )

    caveats: list[str] = []
    if running_suites:
        caveats.append(
            "{count} suite(s) are not completed yet; observed coverage is {observed}/{expected} ({coverage:.2%}).".format(
                count=len(running_suites),
                observed=observed_runs,
                expected=expected_runs,
                coverage=observed_runs / expected_runs if expected_runs else 0.0,
            )
        )
    if len(budgets) > 1:
        caveats.append(
            "Budgets are mixed across suites: {mix}.".format(
                mix=_mix_text(_counter_rows(budgets)),
            )
        )
    if len(unique_judges) > 1:
        caveats.append(
            "Judge models are mixed across suites: {mix}.".format(
                mix=_mix_text(_counter_rows(unique_judges)),
            )
        )

    return {
        "generated_at": _utc_now(),
        "reports_root": str(reports_root),
        "targets_dir": str(targets_dir),
        "include_running": include_running,
        "summary": {
            "suite_count": len(suites),
            "completed_suite_count": len(completed_suites),
            "incomplete_suite_count": len(running_suites),
            "observed_runs": observed_runs,
            "expected_runs": expected_runs,
            "coverage_rate": observed_runs / expected_runs if expected_runs else 0.0,
            "guesser_model_count": len(model_summary),
            "target_count": len(target_summary),
            "budget_mix": _counter_rows(budgets),
            "judge_model_mix": _counter_rows(unique_judges),
        },
        "key_findings": key_findings,
        "caveats": caveats,
        "suites": suites,
        "model_summary": model_summary,
        "target_summary": target_summary,
        "model_target_summary": model_target_summary,
    }


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def render_cross_suite_report(analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    lines = [
        "# Single-Target Suite Benchmark Analysis",
        "",
        f"- generated_at: {analysis['generated_at']}",
        f"- reports_root: {analysis['reports_root']}",
        f"- suites_included: {summary['suite_count']}",
        f"- completed_suites: {summary['completed_suite_count']}",
        f"- incomplete_suites: {summary['incomplete_suite_count']}",
        f"- observed_runs: {summary['observed_runs']}",
        f"- expected_runs: {summary['expected_runs']}",
        f"- coverage: {_format_percent(summary['coverage_rate'])}",
        f"- guesser_models: {summary['guesser_model_count']}",
        f"- targets: {summary['target_count']}",
        f"- budgets: {_mix_text(summary['budget_mix'])}",
        f"- judge_models: {_mix_text(summary['judge_model_mix'])}",
        "",
        "## Key Findings",
        "",
    ]
    if analysis["key_findings"]:
        for finding in analysis["key_findings"]:
            lines.append(f"- {finding}")
    else:
        lines.append("- No runs were found.")

    lines.extend(
        [
            "",
            "## Model Summary",
            "",
            "| guesser_model | observed/planned | solve_rate | avg_turns | avg_turns_solved | budget_exhaustion | errors | targets |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in analysis["model_summary"]:
        lines.append(
            "| {guesser_model} | {observed_runs}/{expected_runs} | {solve_rate} | {avg_turns} | {avg_turns_solved} | {budget_exhaustion_rate} | {error_runs} | {targets} |".format(
                guesser_model=row["guesser_model"],
                observed_runs=row["observed_runs"],
                expected_runs=row["expected_runs"],
                solve_rate=_format_percent(row["solve_rate"]),
                avg_turns=_format_float(row["avg_turns_used"]),
                avg_turns_solved=_format_float(row["avg_turns_solved"]),
                budget_exhaustion_rate=_format_percent(row["budget_exhaustion_rate"]),
                error_runs=row["error_runs"],
                targets=", ".join(row["targets_covered"]),
            )
        )

    lines.extend(
        [
            "",
            "## Target Summary",
            "",
            "| target_id | name | domain | observed/planned | solve_rate | avg_turns | budget_exhaustion | best_model |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in analysis["target_summary"]:
        lines.append(
            "| {target_id} | {target_name} | {target_domain} | {observed_runs}/{expected_runs} | {solve_rate} | {avg_turns} | {budget_exhaustion_rate} | {best_model} |".format(
                target_id=row["target_id"],
                target_name=row["target_name"],
                target_domain=row["target_domain"],
                observed_runs=row["observed_runs"],
                expected_runs=row["expected_runs"],
                solve_rate=_format_percent(row["solve_rate"]),
                avg_turns=_format_float(row["avg_turns_used"]),
                budget_exhaustion_rate=_format_percent(row["budget_exhaustion_rate"]),
                best_model=row["best_model"] or "-",
            )
        )

    lines.extend(
        [
            "",
            "## Model x Target",
            "",
            "| target_id | guesser_model | observed/planned | solve_rate | avg_turns | avg_turns_solved | budget_exhaustion | errors | budget_mix | judge_mix |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in analysis["model_target_summary"]:
        lines.append(
            "| {target_id} | {guesser_model} | {observed_runs}/{expected_runs} | {solve_rate} | {avg_turns} | {avg_turns_solved} | {budget_exhaustion_rate} | {error_runs} | {budget_mix} | {judge_mix} |".format(
                target_id=row["target_id"],
                guesser_model=row["guesser_model"],
                observed_runs=row["observed_runs"],
                expected_runs=row["expected_runs"],
                solve_rate=_format_percent(row["solve_rate"]),
                avg_turns=_format_float(row["avg_turns_used"]),
                avg_turns_solved=_format_float(row["avg_turns_solved"]),
                budget_exhaustion_rate=_format_percent(row["budget_exhaustion_rate"]),
                error_runs=row["error_runs"],
                budget_mix=_mix_text(row["budget_mix"]),
                judge_mix=_mix_text(row["judge_model_mix"]),
            )
        )

    lines.extend(
        [
            "",
            "## Suite Coverage",
            "",
            "| suite_id | suite_name | status | budget | observed/planned | targets | variants |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for suite in sorted(analysis["suites"], key=lambda row: row["suite_id"]):
        lines.append(
            "| {suite_id} | {suite_name} | {status} | {budget} | {observed_runs}/{expected_runs} | {targets} | {variants} |".format(
                suite_id=suite["suite_id"],
                suite_name=suite["suite_name"],
                status=suite["status"],
                budget=suite["budget"],
                observed_runs=suite["observed_runs"],
                expected_runs=suite["expected_runs"],
                targets=", ".join(suite["targets"]),
                variants=", ".join(variant["label"] for variant in suite["variants"]),
            )
        )

    lines.extend(["", "## Caveats", ""])
    if analysis["caveats"]:
        for caveat in analysis["caveats"]:
            lines.append(f"- {caveat}")
    else:
        lines.append("- None.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate single-target suite reports across suite directories.")
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Root directory containing per-suite report folders.",
    )
    parser.add_argument(
        "--targets-dir",
        type=Path,
        default=ROOT / "data" / "all_target.csv",
        help="Directory containing target CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write aggregate.json and report.md into.",
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Only include suites whose status.json says completed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = build_cross_suite_analysis(
        args.reports_root,
        targets_dir=args.targets_dir,
        include_running=not args.completed_only,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "aggregate.json", analysis)
    (args.output_dir / "report.md").write_text(render_cross_suite_report(analysis), encoding="utf-8")
    print(
        json.dumps(
            {
                "reports_root": str(args.reports_root),
                "output_dir": str(args.output_dir),
                "observed_runs": analysis["summary"]["observed_runs"],
                "expected_runs": analysis["summary"]["expected_runs"],
                "coverage_rate": analysis["summary"]["coverage_rate"],
                "suite_count": analysis["summary"]["suite_count"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
