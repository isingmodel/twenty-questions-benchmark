from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, pstdev
from typing import Any

from twentyq.data import load_targets
from twentyq.prompts import ROOT


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


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def _penalized_turns(row: dict[str, Any]) -> int:
    turns_used = _safe_int(row.get("turns_used"))
    suite_budget = _safe_int(row.get("suite_budget"))
    if row.get("solved"):
        return turns_used
    if suite_budget > 0:
        return suite_budget
    return turns_used


def _analysis_turn_horizon(rows: list[dict[str, Any]]) -> int:
    solved_turns = [_safe_int(row.get("turns_used")) for row in rows if row.get("solved")]
    if solved_turns:
        return max(solved_turns)
    return max((_safe_int(row.get("turns_used")) for row in rows), default=0)


def _turns_capped_at_horizon(row: dict[str, Any], horizon: int) -> int:
    if horizon <= 0:
        return _safe_int(row.get("turns_used"))
    return min(_safe_int(row.get("turns_used")), horizon)


def _run_is_uncensored_at_horizon(row: dict[str, Any], horizon: int) -> bool:
    return bool(row.get("solved")) or _safe_int(row.get("turns_used")) >= horizon


def _solve_curve_auc_at_horizon(rows: list[dict[str, Any]], horizon: int) -> float:
    if not rows or horizon <= 0:
        return 0.0
    solved_turns = [
        _safe_int(row.get("turns_used"))
        for row in rows
        if row.get("solved") and _safe_int(row.get("turns_used")) <= horizon
    ]
    solved_area = sum(horizon - turn + 1 for turn in solved_turns)
    return solved_area / (len(rows) * horizon)


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
    analysis_turn_horizon: int,
) -> dict[str, Any]:
    solved_rows = [row for row in rows if row.get("solved")]
    error_rows = [row for row in rows if row.get("error")]
    turns = [_safe_int(row.get("turns_used")) for row in rows]
    solved_turns = [_safe_int(row.get("turns_used")) for row in solved_rows]
    penalized_turns = [_penalized_turns(row) for row in rows]
    horizon_capped_turns = [_turns_capped_at_horizon(row, analysis_turn_horizon) for row in rows]
    uncensored_rows = [row for row in rows if _run_is_uncensored_at_horizon(row, analysis_turn_horizon)]
    censored_rows = [row for row in rows if not _run_is_uncensored_at_horizon(row, analysis_turn_horizon)]
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
        "avg_penalized_turns": sum(penalized_turns) / len(penalized_turns) if penalized_turns else 0.0,
        "analysis_turn_horizon": analysis_turn_horizon,
        "avg_turns_capped_at_horizon": sum(horizon_capped_turns) / len(horizon_capped_turns) if horizon_capped_turns else 0.0,
        "turns_per_success_at_horizon": (
            sum(horizon_capped_turns) / len(solved_rows)
            if horizon_capped_turns and solved_rows
            else None
        ),
        "solve_curve_auc_at_horizon": _solve_curve_auc_at_horizon(rows, analysis_turn_horizon),
        "uncensored_by_horizon_runs": len(uncensored_rows),
        "uncensored_by_horizon_rate": len(uncensored_rows) / len(rows) if rows else 0.0,
        "censored_before_horizon_runs": len(censored_rows),
        "censored_before_horizon_rate": len(censored_rows) / len(rows) if rows else 0.0,
        # Backward-compatible aliases for older downstream consumers.
        "horizon_covered_runs": len(uncensored_rows),
        "horizon_coverage_rate": len(uncensored_rows) / len(rows) if rows else 0.0,
        "median_turns_used": median(turns) if turns else 0.0,
        "avg_turns_solved": sum(solved_turns) / len(solved_turns) if solved_turns else None,
        "expected_turns_to_solve": (
            sum(penalized_turns) / len(solved_rows)
            if penalized_turns and solved_rows
            else None
        ),
        "solve_efficiency": (
            len(solved_rows) / sum(penalized_turns)
            if penalized_turns and sum(penalized_turns) > 0
            else 0.0
        ),
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


def _read_suite_input(suite_dir: Path, *, include_running: bool) -> SuiteInput | None:
    manifest_path = suite_dir / "manifest.json"
    results_path = suite_dir / "results.json"
    if not manifest_path.exists() or not results_path.exists():
        return None
    manifest = _read_json(manifest_path)
    results_payload = _read_json(results_path)
    status_path = suite_dir / "status.json"
    status = _read_json(status_path) if status_path.exists() else {}
    suite_status = str(status.get("status", "unknown"))
    if suite_status != "completed" and not include_running:
        return None
    raw_results = results_payload.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError(f"Expected list field 'results' in {results_path}")
    return SuiteInput(
        suite_dir=suite_dir,
        manifest=manifest,
        results=[row for row in raw_results if isinstance(row, dict)],
        status=status,
    )


def _suite_inputs(
    reports_root: Path,
    *,
    include_running: bool,
    suite_dirs: list[Path] | None = None,
) -> list[SuiteInput]:
    suites: list[SuiteInput] = []
    candidate_dirs = suite_dirs
    if candidate_dirs is None:
        candidate_dirs = sorted(path for path in reports_root.iterdir() if path.is_dir())

    for suite_dir in candidate_dirs:
        suite_input = _read_suite_input(suite_dir, include_running=include_running)
        if suite_input is None:
            continue
        suites.append(suite_input)
    return suites


def build_cross_suite_analysis(
    reports_root: Path,
    *,
    targets_dir: Path,
    suite_dirs: list[Path] | None = None,
    include_running: bool = True,
) -> dict[str, Any]:
    targets = load_targets(targets_dir)
    resolved_suite_dirs = [path.resolve() for path in suite_dirs] if suite_dirs else None
    suite_inputs = _suite_inputs(reports_root, include_running=include_running, suite_dirs=resolved_suite_dirs)

    all_rows: list[dict[str, Any]] = []
    expected_by_model: defaultdict[str, int] = defaultdict(int)
    expected_by_target: defaultdict[str, int] = defaultdict(int)
    expected_by_model_target: defaultdict[tuple[str, str], int] = defaultdict(int)
    expected_by_model_repetition: defaultdict[tuple[str, int], int] = defaultdict(int)

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
            for repetition_index in range(1, repetitions + 1):
                expected_by_model_repetition[(guesser_model, repetition_index)] += len(target_ids)

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

    analysis_turn_horizon = _analysis_turn_horizon(all_rows)

    model_summary: list[dict[str, Any]] = []
    for guesser_model in sorted(expected_by_model):
        rows = [row for row in all_rows if row.get("guesser_model") == guesser_model]
        summary = _summarize_runs(
            rows,
            expected_runs=expected_by_model[guesser_model],
            analysis_turn_horizon=analysis_turn_horizon,
        )
        summary.update(
            {
                "guesser_model": guesser_model,
                "targets_covered": sorted({str(row.get("target_id")) for row in rows}),
            }
        )
        model_summary.append(summary)

    target_summary: list[dict[str, Any]] = []
    for target_id in sorted(expected_by_target):
        rows = [row for row in all_rows if row.get("target_id") == target_id]
        summary = _summarize_runs(
            rows,
            expected_runs=expected_by_target[target_id],
            analysis_turn_horizon=analysis_turn_horizon,
        )
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
        summary = _summarize_runs(
            rows,
            expected_runs=expected_by_model_target[(guesser_model, target_id)],
            analysis_turn_horizon=analysis_turn_horizon,
        )
        summary.update(
            {
                "guesser_model": guesser_model,
                "target_id": target_id,
                "target_name": targets.get(target_id, {}).get("name", target_id),
                "target_domain": targets.get(target_id, {}).get("domain"),
            }
        )
        model_target_summary.append(summary)

    model_repetition_summary: list[dict[str, Any]] = []
    for guesser_model, repetition_index in sorted(expected_by_model_repetition):
        rows = [
            row
            for row in all_rows
            if row.get("guesser_model") == guesser_model and _safe_int(row.get("repetition_index")) == repetition_index
        ]
        summary = _summarize_runs(
            rows,
            expected_runs=expected_by_model_repetition[(guesser_model, repetition_index)],
            analysis_turn_horizon=analysis_turn_horizon,
        )
        summary.update(
            {
                "guesser_model": guesser_model,
                "repetition_index": repetition_index,
            }
        )
        model_repetition_summary.append(summary)

    repetition_rows_by_model: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_repetition_summary:
        repetition_rows_by_model[str(row["guesser_model"])].append(row)

    for row in model_summary:
        repetition_rows = repetition_rows_by_model[row["guesser_model"]]
        repetition_solve_rates = [bucket["solve_rate"] for bucket in repetition_rows]
        repetition_avg_turns = [bucket["avg_turns_used"] for bucket in repetition_rows]
        repetition_avg_turns_solved = [
            bucket["avg_turns_solved"]
            for bucket in repetition_rows
            if bucket["avg_turns_solved"] is not None
        ]
        repetition_turns_per_success_at_horizon = [
            bucket["turns_per_success_at_horizon"]
            for bucket in repetition_rows
            if bucket["turns_per_success_at_horizon"] is not None
        ]
        repetition_solve_curve_auc_at_horizon = [
            bucket["solve_curve_auc_at_horizon"]
            for bucket in repetition_rows
        ]
        repetition_expected_turns_to_solve = [
            bucket["expected_turns_to_solve"]
            for bucket in repetition_rows
            if bucket["expected_turns_to_solve"] is not None
        ]
        row.update(
            {
                "repetition_bucket_count": len(repetition_rows),
                "repetition_solve_rate_stddev": _stddev([float(value) for value in repetition_solve_rates]),
                "repetition_avg_turns_used_stddev": _stddev([float(value) for value in repetition_avg_turns]),
                "repetition_avg_turns_solved_stddev": (
                    _stddev([float(value) for value in repetition_avg_turns_solved])
                    if repetition_avg_turns_solved
                    else None
                ),
                "repetition_turns_per_success_at_horizon_stddev": (
                    _stddev([float(value) for value in repetition_turns_per_success_at_horizon])
                    if repetition_turns_per_success_at_horizon
                    else None
                ),
                "repetition_solve_curve_auc_at_horizon_stddev": _stddev(
                    [float(value) for value in repetition_solve_curve_auc_at_horizon]
                ),
                "repetition_expected_turns_to_solve_stddev": (
                    _stddev([float(value) for value in repetition_expected_turns_to_solve])
                    if repetition_expected_turns_to_solve
                    else None
                ),
                "repetition_solve_rate_min": min(repetition_solve_rates) if repetition_solve_rates else None,
                "repetition_solve_rate_max": max(repetition_solve_rates) if repetition_solve_rates else None,
            }
        )

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
                -row["solve_curve_auc_at_horizon"],
                row["turns_per_success_at_horizon"] if row["turns_per_success_at_horizon"] is not None else float("inf"),
                -row["solve_rate"],
                -row["coverage_rate"],
                row["avg_turns_used"],
                row["guesser_model"],
            ),
        )[0]
        best_by_target[target_id] = best_row["guesser_model"]

    for target_row in target_summary:
        target_row["best_model"] = best_by_target.get(target_row["target_id"])

    model_summary.sort(
        key=lambda row: (
            -row["solve_curve_auc_at_horizon"],
            row["turns_per_success_at_horizon"] if row["turns_per_success_at_horizon"] is not None else float("inf"),
            -row["solve_rate"],
            row["guesser_model"],
        )
    )
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
        best_model = sorted(
            model_summary,
            key=lambda row: (
                -row["solve_curve_auc_at_horizon"],
                row["turns_per_success_at_horizon"] if row["turns_per_success_at_horizon"] is not None else float("inf"),
                row["guesser_model"],
            ),
        )[0]
        key_findings.append(
            "`{model}` has the best integrated solve curve at AUC {auc:.3f} over the shared {horizon}-turn observed solve horizon.".format(
                model=best_model["guesser_model"],
                auc=best_model["solve_curve_auc_at_horizon"],
                horizon=analysis_turn_horizon,
            )
        )
        fastest_model = sorted(
            [row for row in model_summary if row["turns_per_success_at_horizon"] is not None],
            key=lambda row: (row["turns_per_success_at_horizon"], -row["solve_curve_auc_at_horizon"], row["guesser_model"]),
        )[0]
        key_findings.append(
            "`{model}` has the best horizon-capped efficiency at {turns:.2f} turns per success, using the shared {horizon}-turn observed solve horizon.".format(
                model=fastest_model["guesser_model"],
                turns=fastest_model["turns_per_success_at_horizon"],
                horizon=analysis_turn_horizon,
            )
        )
        most_stable_model = sorted(
            model_summary,
            key=lambda row: (
                row["repetition_turns_per_success_at_horizon_stddev"] if row["repetition_turns_per_success_at_horizon_stddev"] is not None else float("inf"),
                -row["observed_runs"],
                row["guesser_model"],
            ),
        )[0]
        key_findings.append(
            "`{model}` is the most repetition-stable by turns-per-success, with a stddev of {stddev:.2f} across repetition buckets.".format(
                model=most_stable_model["guesser_model"],
                stddev=most_stable_model["repetition_turns_per_success_at_horizon_stddev"] or 0.0,
            )
        )
        most_uncensored_model = sorted(
            model_summary,
            key=lambda row: (-row["uncensored_by_horizon_rate"], -row["observed_runs"], row["guesser_model"]),
        )[0]
        key_findings.append(
            "`{model}` has the least right-censoring at the shared horizon, with {rate:.2%} of runs remaining observable through turn {horizon}.".format(
                model=most_uncensored_model["guesser_model"],
                rate=most_uncensored_model["uncensored_by_horizon_rate"],
                horizon=analysis_turn_horizon,
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
    undercovered_pairs = [row for row in model_target_summary if row["coverage_rate"] < 1.0]
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
        caveats.append("Budgets are mixed across suites: {mix}.".format(mix=_mix_text(_counter_rows(budgets))))
    if len(unique_judges) > 1:
        caveats.append(
            "Judge models are mixed across suites: {mix}.".format(
                mix=_mix_text(_counter_rows(unique_judges)),
            )
        )
    high_censoring_models = [row for row in model_summary if row["censored_before_horizon_rate"] > 0.25]
    if high_censoring_models:
        most_censored = sorted(
            high_censoring_models,
            key=lambda row: (-row["censored_before_horizon_rate"], row["guesser_model"]),
        )[0]
        caveats.append(
            "`{model}` has {rate:.2%} right-censored runs before the shared {horizon}-turn horizon, so horizon-based comparisons are less stable for that model.".format(
                model=most_censored["guesser_model"],
                rate=most_censored["censored_before_horizon_rate"],
                horizon=analysis_turn_horizon,
            )
        )

    return {
        "generated_at": _utc_now(),
        "reports_root": str(reports_root),
        "selected_suite_dirs": [str(path) for path in resolved_suite_dirs] if resolved_suite_dirs else None,
        "targets_dir": str(targets_dir),
        "include_running": include_running,
        "summary": {
            "analysis_turn_horizon": analysis_turn_horizon,
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
        "model_repetition_summary": model_repetition_summary,
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
        f"- selected_suite_dirs: {', '.join(analysis['selected_suite_dirs']) if analysis.get('selected_suite_dirs') else '(all suites under reports_root)'}",
        f"- analysis_turn_horizon: {summary['analysis_turn_horizon']}",
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
            "## Metric Notes",
            "",
            "- `analysis_turn_horizon` is the largest turn at which any solve was observed in the included runs. It is derived from the data, not from the configured suite budget.",
            "- `solve_curve_auc` is the area under the empirical solve-by-turn curve from turn 1 through the shared horizon. Higher is better.",
            "- `turns_per_success_h` is `sum(min(turns_used, horizon)) / solved_runs`. It is a horizon-capped cost-per-success metric, so unsolved runs consume turns up to the shared horizon instead of the configured budget.",
            "- `censored_before_h` is the share of runs that stopped unsolved before the shared horizon. Lower is better, because fewer runs are right-censored before the comparison window ends.",
            "",
            "## Model Summary",
            "",
            "| guesser_model | observed/planned | solve_rate | solve_curve_auc | turns_per_success_h | avg_turns_capped_h | censored_before_h | avg_turns | errors | targets |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in analysis["model_summary"]:
        lines.append(
            "| {guesser_model} | {observed_runs}/{expected_runs} | {solve_rate} | {solve_curve_auc} | {turns_per_success_h} | {avg_turns_capped_h} | {censored_before_h} | {avg_turns} | {error_runs} | {targets} |".format(
                guesser_model=row["guesser_model"],
                observed_runs=row["observed_runs"],
                expected_runs=row["expected_runs"],
                solve_rate=_format_percent(row["solve_rate"]),
                solve_curve_auc=_format_float(row["solve_curve_auc_at_horizon"]),
                turns_per_success_h=_format_float(row["turns_per_success_at_horizon"]),
                avg_turns_capped_h=_format_float(row["avg_turns_capped_at_horizon"]),
                censored_before_h=_format_percent(row["censored_before_horizon_rate"]),
                avg_turns=_format_float(row["avg_turns_used"]),
                error_runs=row["error_runs"],
                targets=", ".join(row["targets_covered"]),
            )
        )

    lines.extend(
        [
            "",
            "## Repetition Stability",
            "",
            "| guesser_model | repetition_buckets | solve_rate_stddev | solve_curve_auc_stddev | turns_per_success_h_stddev | avg_turns_stddev | solve_rate_range | censoring |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in analysis["model_summary"]:
        solve_rate_range = "-"
        if row.get("repetition_solve_rate_min") is not None and row.get("repetition_solve_rate_max") is not None:
            solve_rate_range = "{min_rate} - {max_rate}".format(
                min_rate=_format_percent(float(row["repetition_solve_rate_min"])),
                max_rate=_format_percent(float(row["repetition_solve_rate_max"])),
            )
        lines.append(
            "| {guesser_model} | {repetition_bucket_count} | {solve_rate_stddev} | {solve_curve_auc_stddev} | {turns_per_success_h_stddev} | {avg_turns_stddev} | {solve_rate_range} | {censoring} |".format(
                guesser_model=row["guesser_model"],
                repetition_bucket_count=row["repetition_bucket_count"],
                solve_rate_stddev=_format_float(row["repetition_solve_rate_stddev"]),
                solve_curve_auc_stddev=_format_float(row["repetition_solve_curve_auc_at_horizon_stddev"]),
                turns_per_success_h_stddev=_format_float(row["repetition_turns_per_success_at_horizon_stddev"]),
                avg_turns_stddev=_format_float(row["repetition_avg_turns_used_stddev"]),
                solve_rate_range=solve_rate_range,
                censoring=_format_percent(row["censored_before_horizon_rate"]),
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
            "| target_id | guesser_model | observed/planned | solve_rate | solve_curve_auc | turns_per_success_h | censored_before_h | avg_turns | errors | budget_mix | judge_mix |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in analysis["model_target_summary"]:
        lines.append(
            "| {target_id} | {guesser_model} | {observed_runs}/{expected_runs} | {solve_rate} | {solve_curve_auc} | {turns_per_success_h} | {censored_before_h} | {avg_turns} | {error_runs} | {budget_mix} | {judge_mix} |".format(
                target_id=row["target_id"],
                guesser_model=row["guesser_model"],
                observed_runs=row["observed_runs"],
                expected_runs=row["expected_runs"],
                solve_rate=_format_percent(row["solve_rate"]),
                solve_curve_auc=_format_float(row["solve_curve_auc_at_horizon"]),
                turns_per_success_h=_format_float(row["turns_per_success_at_horizon"]),
                censored_before_h=_format_percent(row["censored_before_horizon_rate"]),
                avg_turns=_format_float(row["avg_turns_used"]),
                error_runs=row["error_runs"],
                budget_mix=_mix_text(row["budget_mix"]),
                judge_mix=_mix_text(row["judge_model_mix"]),
            )
        )

    lines.extend(
        [
            "",
            "## Model x Repetition",
            "",
            "| repetition_index | guesser_model | observed/planned | solve_rate | solve_curve_auc | turns_per_success_h | censored_before_h | avg_turns | errors |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["model_repetition_summary"]:
        lines.append(
            "| {repetition_index} | {guesser_model} | {observed_runs}/{expected_runs} | {solve_rate} | {solve_curve_auc} | {turns_per_success_h} | {censored_before_h} | {avg_turns} | {error_runs} |".format(
                repetition_index=row["repetition_index"],
                guesser_model=row["guesser_model"],
                observed_runs=row["observed_runs"],
                expected_runs=row["expected_runs"],
                solve_rate=_format_percent(row["solve_rate"]),
                solve_curve_auc=_format_float(row["solve_curve_auc_at_horizon"]),
                turns_per_success_h=_format_float(row["turns_per_success_at_horizon"]),
                censored_before_h=_format_percent(row["censored_before_horizon_rate"]),
                avg_turns=_format_float(row["avg_turns_used"]),
                error_runs=row["error_runs"],
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
    parser.add_argument(
        "--suite-dir",
        action="append",
        type=Path,
        default=None,
        help="Optional explicit suite directory to include. Repeat to merge specific suites only.",
    )
    parser.add_argument(
        "--results-json",
        action="append",
        type=Path,
        default=None,
        help="Optional path to a suite results.json file. Repeat to merge specific suites by results artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_suite_dirs: list[Path] | None = None
    if args.suite_dir or args.results_json:
        selected_suite_dirs = []
        if args.suite_dir:
            selected_suite_dirs.extend(args.suite_dir)
        if args.results_json:
            selected_suite_dirs.extend(path.parent for path in args.results_json)
    analysis = build_cross_suite_analysis(
        args.reports_root,
        targets_dir=args.targets_dir,
        suite_dirs=selected_suite_dirs,
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
