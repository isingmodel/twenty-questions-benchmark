from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .data import load_data
from .env import load_dotenv
from .episode_runner import (
    DEFAULT_GUESSER_MODEL,
    DEFAULT_JUDGE_MODEL,
    THINKING_LEVEL_CHOICES,
    FullGameConfig,
    _validate_budget,
    _validate_reasoning_config,
    provider_for_model,
    run_full_game_episode,
)
from .prompts import ROOT



DEFAULT_BUDGET = 80


@dataclass
class BenchmarkConfig:

    budget: int
    guesser_model: str
    judge_model: str
    guesser_thinking_level: str | None
    judge_thinking_level: str | None
    guesser_thinking_budget: int | None
    judge_thinking_budget: int | None
    benchmark_dir: Path | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_benchmark_dir(config: BenchmarkConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = config.guesser_model.replace("/", "-")
    guesser_provider = provider_for_model(config.guesser_model)
    unique_suffix = uuid4().hex[:8]
    return (
        ROOT
        / "reports"
        / f"{guesser_provider}-benchmark"
        / f"{stamp}__{guesser_provider}__budget{config.budget}__{slug}__{unique_suffix}"
    )


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run a sequential benchmark across supported providers.")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Maximum turns per target.")
    parser.add_argument("--guesser-model", default=DEFAULT_GUESSER_MODEL, help="Guesser model id.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model id.")
    parser.add_argument(
        "--guesser-thinking-level",
        choices=THINKING_LEVEL_CHOICES,
        default=None,
        help="Reasoning level for level-based models (for example Gemini 3 or OpenAI effort mapping).",
    )
    parser.add_argument(
        "--judge-thinking-level",
        choices=THINKING_LEVEL_CHOICES,
        default=None,
        help="Reasoning level for level-based models (for example Gemini 3 or OpenAI effort mapping).",
    )
    parser.add_argument(
        "--guesser-thinking-budget",
        type=int,
        default=None,
        help="Thinking budget for budget-based models (for example Gemini 2.5 or Claude thinking models).",
    )
    parser.add_argument(
        "--judge-thinking-budget",
        type=int,
        default=None,
        help="Thinking budget for budget-based models (for example Gemini 2.5 or Claude thinking models).",
    )
    parser.add_argument("--benchmark-dir", type=Path, default=None, help="Optional benchmark output directory.")
    args = parser.parse_args()
    return BenchmarkConfig(
        budget=_validate_budget(args.budget),
        guesser_model=args.guesser_model,
        judge_model=args.judge_model,
        guesser_thinking_level=args.guesser_thinking_level,
        judge_thinking_level=args.judge_thinking_level,
        guesser_thinking_budget=args.guesser_thinking_budget,
        judge_thinking_budget=args.judge_thinking_budget,
        benchmark_dir=args.benchmark_dir,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _initial_status(config: BenchmarkConfig, targets: list[dict[str, Any]], benchmark_dir: Path) -> dict[str, Any]:
    guesser_provider = provider_for_model(config.guesser_model)
    judge_provider = provider_for_model(config.judge_model)
    return {
        "benchmark_dir": str(benchmark_dir),
        "status": "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed_at": None,
        "budget": config.budget,
        "guesser_provider": guesser_provider,
        "guesser_model": config.guesser_model,
        "judge_provider": judge_provider,
        "judge_model": config.judge_model,
        "targets_total": len(targets),
        "targets_completed": 0,
        "targets_failed": 0,
        "targets_transient_errors": 0,
        "current_target_id": None,
        "current_run_id": None,
        "current_turn": 0,
        "current_question": None,
        "current_judgment": None,
        "results_path": str(benchmark_dir / "benchmark_results.json"),
        "runs_dir": str(benchmark_dir / "runs"),
        "targets": [
            {"target_id": target["id"], "status": "pending", "run_id": None, "summary_path": None, "error": None}
            for target in targets
        ],
    }


def _aggregate(config: BenchmarkConfig, benchmark_dir: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    guesser_provider = provider_for_model(config.guesser_model)
    judge_provider = provider_for_model(config.judge_model)
    solved_results = [result for result in results if result.get("solved")]
    transient_error_results = [result for result in results if result.get("error_type") == "transient_error"]
    failed_results = [result for result in results if result.get("error") and result.get("error_type") != "transient_error"]
    turns = [int(result.get("turns_used", 0)) for result in results]
    return {
        "benchmark_dir": str(benchmark_dir),
        "completed_at": _utc_now(),
        "budget": config.budget,
        "guesser_provider": guesser_provider,
        "guesser_model": config.guesser_model,
        "judge_provider": judge_provider,
        "judge_model": config.judge_model,
        "targets_total": len(results),
        "targets_solved": len(solved_results),
        "targets_failed": len(failed_results),
        "targets_transient_errors": len(transient_error_results),
        "solve_rate": len(solved_results) / len(results) if results else 0.0,
        "avg_turns_used": sum(turns) / len(turns) if turns else 0.0,
        "results": results,
    }


def main() -> int:
    config = parse_args()
    load_dotenv(ROOT / ".env")
    benchmark_dir = config.benchmark_dir or _default_benchmark_dir(config)
    data_path = ROOT / "data" / "all_target.csv"
    targets = load_data(data_path)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    guesser_reasoning = _validate_reasoning_config(
        role="Guesser",
        model=config.guesser_model,
        thinking_level=config.guesser_thinking_level,
        thinking_budget=config.guesser_thinking_budget,
    )
    judge_reasoning = _validate_reasoning_config(
        role="Judge",
        model=config.judge_model,
        thinking_level=config.judge_thinking_level,
        thinking_budget=config.judge_thinking_budget,
    )

    manifest = {
        "created_at": _utc_now(),
        "mode": "full-game-benchmark",
        "budget": config.budget,
        "guesser_provider": provider_for_model(config.guesser_model),
        "guesser_model": config.guesser_model,
        "judge_provider": provider_for_model(config.judge_model),
        "judge_model": config.judge_model,
        "targets": [target["id"] for target in targets],
    }
    _write_json(benchmark_dir / "manifest.json", manifest)

    status = _initial_status(config, targets, benchmark_dir)
    _write_json(benchmark_dir / "status.json", status)

    results: list[dict[str, Any]] = []
    runs_dir = benchmark_dir / "runs"
    for index, target in enumerate(targets):
        status["current_target_id"] = target["id"]
        status["current_run_id"] = None
        status["current_turn"] = 0
        status["current_question"] = None
        status["current_judgment"] = None
        status["updated_at"] = _utc_now()
        status["targets"][index]["status"] = "running"
        _write_json(benchmark_dir / "status.json", status)

        def _progress(event: dict[str, Any]) -> None:
            status["updated_at"] = _utc_now()
            status["current_run_id"] = event.get("run_id", status["current_run_id"])
            status["current_target_id"] = event.get("target_id", status["current_target_id"])
            status["current_turn"] = event.get("turn", status["current_turn"])
            if "question" in event:
                status["current_question"] = event.get("question")
            if "judgment" in event:
                status["current_judgment"] = event.get("judgment")
            _write_json(benchmark_dir / "status.json", status)

        exit_code, summary = run_full_game_episode(
            config=FullGameConfig(
                target_id=target["id"],
                budget=config.budget,
                guesser_model=config.guesser_model,
                judge_model=config.judge_model,
                guesser_reasoning=guesser_reasoning,
                judge_reasoning=judge_reasoning,
                run_dir=runs_dir,
            ),
            target=target,
            runs_dir=runs_dir,
            progress_callback=_progress,
        )
        results.append(summary)
        status["targets_completed"] += 1
        status["updated_at"] = _utc_now()
        status["targets"][index]["run_id"] = summary["run_id"]
        status["targets"][index]["summary_path"] = f"{summary['run_dir']}/summary.json"
        if exit_code == 0:
            status["targets"][index]["status"] = "completed"
        else:
            error_type = summary.get("error_type", "runtime_error")
            if error_type == "transient_error":
                status["targets_transient_errors"] += 1
                status["targets"][index]["status"] = "transient_error"
            else:
                status["targets_failed"] += 1
                status["targets"][index]["status"] = "failed"
            status["targets"][index]["error"] = summary.get("error")
            status["targets"][index]["error_type"] = error_type
        _write_json(benchmark_dir / "benchmark_results.json", {"results": results})
        _write_json(benchmark_dir / "status.json", status)

    aggregate = _aggregate(config, benchmark_dir, results)
    _write_json(benchmark_dir / "aggregate.json", aggregate)
    status["status"] = "completed"
    status["updated_at"] = aggregate["completed_at"]
    status["completed_at"] = aggregate["completed_at"]
    status["current_target_id"] = None
    status["current_run_id"] = None
    status["current_turn"] = 0
    status["current_question"] = None
    status["current_judgment"] = None
    _write_json(benchmark_dir / "status.json", status)
    print(json.dumps(aggregate, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
