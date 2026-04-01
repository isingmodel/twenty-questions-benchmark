from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .data import load_targets
from .env import load_dotenv
from .episode_runner import (
    PROVIDER,
    FullGameConfig,
    _reasoning_to_payload,
    _validate_budget,
    resolve_reasoning_effort,
    run_full_game_episode,
)
from .prompts import ROOT


DEFAULT_OUTPUT_PARENT = ROOT / "reports" / "gemini-single-target-suite"


@dataclass(frozen=True)
class ModelVariant:
    label: str
    guesser_model: str
    guesser_reasoning_effort: str | None
    judge_model: str
    judge_reasoning_effort: str | None
    repetitions: int


@dataclass(frozen=True)
class SingleTargetSuiteConfig:
    suite_name: str
    target_ids: tuple[str, ...]
    budget: int
    variants: tuple[ModelVariant, ...]
    output_dir: Path | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _sanitize_fragment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip().lower())
    return cleaned.strip("-")


def _default_suite_dir(config: SingleTargetSuiteConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_slug = "-".join(_sanitize_fragment(target_id) for target_id in config.target_ids)
    return DEFAULT_OUTPUT_PARENT / f"{stamp}__{_sanitize_fragment(config.suite_name)}__{target_slug}__budget{config.budget}"


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string field {field_name!r}")
    return value.strip()


def _require_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected non-empty list field {field_name!r}")
    normalized: list[str] = []
    for item in value:
        normalized.append(_require_string(item, field_name))
    return tuple(normalized)


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"Expected positive integer field {field_name!r}")
    return value


def load_suite_config(config_path: Path) -> SingleTargetSuiteConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    suite_name = _require_string(raw.get("suite_name"), "suite_name")
    target_ids = _require_string_list(raw.get("targets"), "targets")
    budget = _validate_budget(_require_positive_int(raw.get("budget"), "budget"))
    default_repetitions = _require_positive_int(raw.get("repetitions"), "repetitions")
    default_judge_model = _require_string(raw.get("judge_model"), "judge_model")
    default_guesser_effort = raw.get("guesser_reasoning_effort")
    default_judge_effort = raw.get("judge_reasoning_effort")
    if default_guesser_effort is not None:
        default_guesser_effort = _require_string(default_guesser_effort, "guesser_reasoning_effort")
    if default_judge_effort is not None:
        default_judge_effort = _require_string(default_judge_effort, "judge_reasoning_effort")

    raw_variants = raw.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("Expected non-empty list field 'variants'")

    variants: list[ModelVariant] = []
    for index, raw_variant in enumerate(raw_variants, start=1):
        if not isinstance(raw_variant, dict):
            raise ValueError(f"Expected object for variants[{index - 1}]")
        label = _require_string(raw_variant.get("label"), f"variants[{index - 1}].label")
        guesser_model = _require_string(raw_variant.get("guesser_model"), f"variants[{index - 1}].guesser_model")
        judge_model = raw_variant.get("judge_model", default_judge_model)
        guesser_effort = raw_variant.get("guesser_reasoning_effort", default_guesser_effort)
        judge_effort = raw_variant.get("judge_reasoning_effort", default_judge_effort)
        repetitions = raw_variant.get("repetitions", default_repetitions)
        if judge_model is None:
            raise ValueError(f"Missing judge_model for variants[{index - 1}]")
        if guesser_effort is not None:
            guesser_effort = _require_string(guesser_effort, f"variants[{index - 1}].guesser_reasoning_effort")
        if judge_effort is not None:
            judge_effort = _require_string(judge_effort, f"variants[{index - 1}].judge_reasoning_effort")
        variants.append(
            ModelVariant(
                label=label,
                guesser_model=guesser_model,
                guesser_reasoning_effort=guesser_effort,
                judge_model=_require_string(judge_model, f"variants[{index - 1}].judge_model"),
                judge_reasoning_effort=judge_effort,
                repetitions=_require_positive_int(repetitions, f"variants[{index - 1}].repetitions"),
            )
        )

    output_dir = raw.get("output_dir")
    resolved_output_dir = Path(output_dir) if isinstance(output_dir, str) and output_dir.strip() else None
    return SingleTargetSuiteConfig(
        suite_name=suite_name,
        target_ids=target_ids,
        budget=budget,
        variants=tuple(variants),
        output_dir=resolved_output_dir,
    )


def _initial_status(config: SingleTargetSuiteConfig, suite_dir: Path) -> dict[str, Any]:
    total_runs = sum(len(config.target_ids) * variant.repetitions for variant in config.variants)
    return {
        "suite_name": config.suite_name,
        "suite_dir": str(suite_dir),
        "status": "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed_at": None,
        "budget": config.budget,
        "targets": list(config.target_ids),
        "total_runs": total_runs,
        "runs_completed": 0,
        "runs_failed": 0,
        "runs_transient_errors": 0,
        "current_target_id": None,
        "current_variant_label": None,
        "current_guesser_model": None,
        "current_judge_model": None,
        "current_repetition": None,
        "active_runs": [],
        "results_path": str(suite_dir / "results.json"),
        "aggregate_path": str(suite_dir / "aggregate.json"),
        "report_path": str(suite_dir / "report.md"),
        "runs_dir": str(suite_dir / "runs"),
    }


def _result_record(
    target_id: str,
    variant: ModelVariant,
    repetition_index: int,
    guesser_reasoning: dict[str, Any],
    judge_reasoning: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "variant_label": variant.label,
        "repetition_index": repetition_index,
        "guesser_model": variant.guesser_model,
        "judge_model": variant.judge_model,
        "guesser_reasoning_effort": variant.guesser_reasoning_effort,
        "judge_reasoning_effort": variant.judge_reasoning_effort,
        "guesser_reasoning": guesser_reasoning,
        "judge_reasoning": judge_reasoning,
        **summary,
    }


def aggregate_results(config: SingleTargetSuiteConfig, results: list[dict[str, Any]], suite_dir: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (str(result["target_id"]), str(result["variant_label"]))
        grouped.setdefault(key, []).append(result)

    groups: list[dict[str, Any]] = []
    for target_id, variant_label in sorted(grouped):
        items = grouped[(target_id, variant_label)]
        solved = [item for item in items if item.get("solved")]
        failed = [item for item in items if item.get("error")]
        transient_errors = [item for item in items if item.get("error_type") == "transient_error"]
        turns = [int(item.get("turns_used", 0)) for item in items]
        solved_turns = [int(item.get("turns_used", 0)) for item in solved]
        exemplar = items[0]
        groups.append(
            {
                "target_id": target_id,
                "variant_label": variant_label,
                "guesser_model": exemplar["guesser_model"],
                "judge_model": exemplar["judge_model"],
                "guesser_reasoning_effort": exemplar.get("guesser_reasoning_effort"),
                "judge_reasoning_effort": exemplar.get("judge_reasoning_effort"),
                "guesser_reasoning": exemplar.get("guesser_reasoning", {}),
                "judge_reasoning": exemplar.get("judge_reasoning", {}),
                "runs_total": len(items),
                "runs_solved": len(solved),
                "runs_failed": len(failed),
                "runs_transient_errors": len(transient_errors),
                "solve_rate": len(solved) / len(items) if items else 0.0,
                "avg_turns_used": sum(turns) / len(turns) if turns else 0.0,
                "median_turns_used": median(turns) if turns else 0.0,
                "avg_turns_solved": sum(solved_turns) / len(solved_turns) if solved_turns else None,
                "min_turns_used": min(turns) if turns else 0,
                "max_turns_used": max(turns) if turns else 0,
            }
        )

    solved_results = [result for result in results if result.get("solved")]
    failed_results = [result for result in results if result.get("error")]
    return {
        "suite_name": config.suite_name,
        "suite_dir": str(suite_dir),
        "completed_at": _utc_now(),
        "budget": config.budget,
        "targets": list(config.target_ids),
        "runs_total": len(results),
        "runs_solved": len(solved_results),
        "runs_failed": len(failed_results),
        "solve_rate": len(solved_results) / len(results) if results else 0.0,
        "groups": groups,
        "results": results,
    }


def render_report(config: SingleTargetSuiteConfig, aggregate: dict[str, Any]) -> str:
    lines = [
        f"# Single-Target Suite Report: {config.suite_name}",
        "",
        f"- completed_at: {aggregate['completed_at']}",
        f"- budget: {config.budget}",
        f"- targets: {', '.join(config.target_ids)}",
        f"- runs_total: {aggregate['runs_total']}",
        f"- runs_solved: {aggregate['runs_solved']}",
        f"- runs_failed: {aggregate['runs_failed']}",
        f"- overall_solve_rate: {aggregate['solve_rate']:.2%}",
        "",
        "## Per Variant",
        "",
        "| target_id | variant | guesser_model | judge_model | runs | solved | solve_rate | avg_turns | median_turns | avg_turns_solved | failures |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in aggregate["groups"]:
        avg_turns_solved = group["avg_turns_solved"]
        avg_turns_solved_text = "-" if avg_turns_solved is None else f"{avg_turns_solved:.2f}"
        lines.append(
            "| {target_id} | {variant_label} | {guesser_model} | {judge_model} | {runs_total} | {runs_solved} | {solve_rate:.2%} | {avg_turns_used:.2f} | {median_turns_used:.2f} | {avg_turns_solved} | {runs_failed} |".format(
                target_id=group["target_id"],
                variant_label=group["variant_label"],
                guesser_model=group["guesser_model"],
                judge_model=group["judge_model"],
                runs_total=group["runs_total"],
                runs_solved=group["runs_solved"],
                solve_rate=group["solve_rate"],
                avg_turns_used=group["avg_turns_used"],
                median_turns_used=group["median_turns_used"],
                avg_turns_solved=avg_turns_solved_text,
                runs_failed=group["runs_failed"],
            )
        )

    lines.extend(["", "## Notes", ""])
    for group in aggregate["groups"]:
        lines.append(
            "- `{label}` resolved to guesser reasoning `{gr}` and judge reasoning `{jr}`.".format(
                label=group["variant_label"],
                gr=json.dumps(group["guesser_reasoning"], ensure_ascii=True, sort_keys=True),
                jr=json.dumps(group["judge_reasoning"], ensure_ascii=True, sort_keys=True),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a repeated single-target model comparison suite.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a suite config JSON file.")
    parser.add_argument("--suite-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--max-parallel", type=int, default=2, help="Maximum number of runs to execute in parallel.")
    return parser.parse_args()


def _run_suite_job(
    *,
    target_id: str,
    target: dict[str, Any],
    variant: ModelVariant,
    repetition_index: int,
    budget: int,
    runs_dir: Path,
) -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    guesser_reasoning = resolve_reasoning_effort(
        variant.guesser_model,
        variant.guesser_reasoning_effort,
        role="Guesser",
    )
    judge_reasoning = resolve_reasoning_effort(
        variant.judge_model,
        variant.judge_reasoning_effort,
        role="Judge",
    )
    exit_code, summary = run_full_game_episode(
        config=FullGameConfig(
            target_id=target_id,
            budget=budget,
            guesser_model=variant.guesser_model,
            judge_model=variant.judge_model,
            guesser_reasoning=guesser_reasoning,
            judge_reasoning=judge_reasoning,
            run_dir=runs_dir,
        ),
        target=target,
        runs_dir=runs_dir,
    )
    return (
        exit_code,
        summary,
        _reasoning_to_payload(guesser_reasoning),
        _reasoning_to_payload(judge_reasoning),
    )


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    config = load_suite_config(args.config)
    if args.max_parallel < 1:
        raise ValueError(f"--max-parallel must be positive, got {args.max_parallel}")
    suite_dir = args.suite_dir or config.output_dir or _default_suite_dir(config)
    suite_dir.mkdir(parents=True, exist_ok=True)

    targets_by_id = load_targets(ROOT / "data" / "targets")
    missing_targets = [target_id for target_id in config.target_ids if target_id not in targets_by_id]
    if missing_targets:
        raise ValueError(f"Unknown target ids in suite config: {missing_targets!r}")

    status = _initial_status(config, suite_dir)
    results: list[dict[str, Any]] = []
    runs_dir = suite_dir / "runs"

    manifest = {
        "created_at": _utc_now(),
        "suite_name": config.suite_name,
        "budget": config.budget,
        "max_parallel": args.max_parallel,
        "targets": list(config.target_ids),
        "variants": [
            {
                "label": variant.label,
                "guesser_model": variant.guesser_model,
                "guesser_reasoning_effort": variant.guesser_reasoning_effort,
                "judge_model": variant.judge_model,
                "judge_reasoning_effort": variant.judge_reasoning_effort,
                "repetitions": variant.repetitions,
                "resolved_guesser_reasoning": _reasoning_to_payload(
                    resolve_reasoning_effort(variant.guesser_model, variant.guesser_reasoning_effort, role="Guesser")
                ),
                "resolved_judge_reasoning": _reasoning_to_payload(
                    resolve_reasoning_effort(variant.judge_model, variant.judge_reasoning_effort, role="Judge")
                ),
            }
            for variant in config.variants
        ],
    }
    _write_json(suite_dir / "manifest.json", manifest)
    _write_json(suite_dir / "status.json", status)
    _write_json(suite_dir / "results.json", {"results": results})

    jobs: list[dict[str, Any]] = []
    task_index = 0
    for target_id in config.target_ids:
        target = targets_by_id[target_id]
        for variant in config.variants:
            for repetition_index in range(1, variant.repetitions + 1):
                jobs.append(
                    {
                        "task_index": task_index,
                        "target_id": target_id,
                        "target": target,
                        "variant": variant,
                        "repetition_index": repetition_index,
                    }
                )
                task_index += 1

    pending_results: dict[int, dict[str, Any]] = {}
    active_runs_by_future: dict[concurrent.futures.Future[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]], dict[str, Any]] = {}
    next_job_index = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        while next_job_index < len(jobs) or active_runs_by_future:
            while next_job_index < len(jobs) and len(active_runs_by_future) < args.max_parallel:
                job = jobs[next_job_index]
                next_job_index += 1
                status["updated_at"] = _utc_now()
                status["current_target_id"] = job["target_id"]
                status["current_variant_label"] = job["variant"].label
                status["current_guesser_model"] = job["variant"].guesser_model
                status["current_judge_model"] = job["variant"].judge_model
                status["current_repetition"] = job["repetition_index"]
                status["active_runs"] = [
                    *status["active_runs"],
                    {
                        "target_id": job["target_id"],
                        "variant_label": job["variant"].label,
                        "guesser_model": job["variant"].guesser_model,
                        "judge_model": job["variant"].judge_model,
                        "repetition_index": job["repetition_index"],
                    },
                ]
                _write_json(suite_dir / "status.json", status)
                future = executor.submit(
                    _run_suite_job,
                    target_id=job["target_id"],
                    target=job["target"],
                    variant=job["variant"],
                    repetition_index=job["repetition_index"],
                    budget=config.budget,
                    runs_dir=runs_dir,
                )
                active_runs_by_future[future] = job

            completed_future = next(concurrent.futures.as_completed(active_runs_by_future))
            job = active_runs_by_future.pop(completed_future)
            exit_code, summary, guesser_reasoning, judge_reasoning = completed_future.result()
            result = _result_record(
                target_id=job["target_id"],
                variant=job["variant"],
                repetition_index=job["repetition_index"],
                guesser_reasoning=guesser_reasoning,
                judge_reasoning=judge_reasoning,
                summary=summary,
            )
            pending_results[job["task_index"]] = result
            results = [pending_results[index] for index in sorted(pending_results)]
            status["runs_completed"] += 1
            if exit_code != 0:
                if summary.get("error_type") == "transient_error":
                    status["runs_transient_errors"] += 1
                else:
                    status["runs_failed"] += 1
            status["active_runs"] = [
                active_run
                for active_run in status["active_runs"]
                if not (
                    active_run["target_id"] == job["target_id"]
                    and active_run["variant_label"] == job["variant"].label
                    and active_run["repetition_index"] == job["repetition_index"]
                )
            ]
            status["updated_at"] = _utc_now()
            _write_json(suite_dir / "results.json", {"results": results})
            _write_json(suite_dir / "status.json", status)

    aggregate = aggregate_results(config, results, suite_dir)
    _write_json(suite_dir / "aggregate.json", aggregate)
    (suite_dir / "report.md").write_text(render_report(config, aggregate), encoding="utf-8")

    status["status"] = "completed"
    status["completed_at"] = aggregate["completed_at"]
    status["updated_at"] = aggregate["completed_at"]
    status["current_target_id"] = None
    status["current_variant_label"] = None
    status["current_guesser_model"] = None
    status["current_judge_model"] = None
    status["current_repetition"] = None
    status["active_runs"] = []
    _write_json(suite_dir / "status.json", status)
    print(
        json.dumps(
            {
                "suite_name": config.suite_name,
                "suite_dir": str(suite_dir),
                "report_path": str(suite_dir / "report.md"),
                "aggregate_path": str(suite_dir / "aggregate.json"),
                "results_path": str(suite_dir / "results.json"),
                "runs_total": aggregate["runs_total"],
                "runs_solved": aggregate["runs_solved"],
                "runs_failed": aggregate["runs_failed"],
                "solve_rate": aggregate["solve_rate"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
