from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "reports" / "all_sessions"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "results_all_sessions.csv"

CSV_FIELDNAMES = [
    "target_id",
    "variant_label",
    "repetition_index",
    "guesser_model",
    "judge_model",
    "guesser_reasoning_effort",
    "judge_reasoning_effort",
    "guesser_reasoning",
    "judge_reasoning",
    "run_id",
    "mode",
    "target_name",
    "solved",
    "turns_used",
    "final_question",
    "final_question_correct",
    "error",
    "error_type",
    "guesser_w_effort",
]

RUN_ID_PATTERN = re.compile(r"^run-(\d+)")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_run_number(run_id: str, run_dir_name: str) -> int:
    for candidate in (run_id, run_dir_name):
        match = RUN_ID_PATTERN.match(candidate)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"Could not extract run number from run_id={run_id!r} run_dir_name={run_dir_name!r}")


def _normalize_reasoning_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _reasoning_effort_label(reasoning: dict[str, Any]) -> str:
    effort = reasoning.get("reasoning_effort")
    if isinstance(effort, str) and effort:
        return effort
    thinking_budget = reasoning.get("thinking_budget")
    if isinstance(thinking_budget, int):
        return f"budget_{thinking_budget}"
    return ""


def _stringify_reasoning(reasoning: dict[str, Any]) -> str:
    return str(reasoning)


def _row_from_run_dir(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    run_config_path = run_dir / "run_config.json"
    if not summary_path.exists() or not run_config_path.exists():
        return None

    summary = _read_json(summary_path)
    run_config = _read_json(run_config_path)
    config = run_config.get("config")
    if not isinstance(config, dict):
        config = {}

    run_id = str(summary.get("run_id") or run_config.get("run_id") or run_dir.name)
    guesser_model = str(config.get("guesser_model") or "")
    judge_model = str(config.get("judge_model") or "")
    guesser_reasoning = _normalize_reasoning_payload(config.get("guesser_reasoning"))
    judge_reasoning = _normalize_reasoning_payload(config.get("judge_reasoning"))
    guesser_reasoning_effort = _reasoning_effort_label(guesser_reasoning)
    judge_reasoning_effort = _reasoning_effort_label(judge_reasoning)

    return {
        "target_id": summary.get("target_id") or config.get("target_id") or "",
        "variant_label": guesser_model,
        "repetition_index": _extract_run_number(run_id, run_dir.name),
        "guesser_model": guesser_model,
        "judge_model": judge_model,
        "guesser_reasoning_effort": guesser_reasoning_effort,
        "judge_reasoning_effort": judge_reasoning_effort,
        "guesser_reasoning": _stringify_reasoning(guesser_reasoning),
        "judge_reasoning": _stringify_reasoning(judge_reasoning),
        "run_id": run_id,
        "mode": summary.get("mode") or run_config.get("mode") or "",
        "target_name": summary.get("target_name") or "",
        "solved": summary.get("solved", False),
        "turns_used": summary.get("turns_used", ""),
        "final_question": summary.get("final_question") or "",
        "final_question_correct": summary.get("final_question_correct", False),
        "error": summary.get("error") or "",
        "error_type": summary.get("error_type") or "",
        "guesser_w_effort": f"{guesser_model}_{guesser_reasoning_effort}" if guesser_reasoning_effort else guesser_model,
    }


def collect_all_session_rows(runs_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        row = _row_from_run_dir(run_dir)
        if row is None:
            skipped.append(run_dir.name)
            continue
        rows.append(row)

    return rows, skipped


def write_results_csv(runs_root: Path, output_path: Path) -> tuple[int, list[str]]:
    rows, skipped = collect_all_session_rows(runs_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), skipped


def update_results_all_sessions_csv(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[int, list[str]]:
    return write_results_csv(runs_root, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a run-level CSV from reports/all_sessions.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Directory containing all-session run folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path to write.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written, skipped = update_results_all_sessions_csv(args.runs_root, args.output)
    print(
        json.dumps(
            {
                "runs_root": str(args.runs_root),
                "output": str(args.output),
                "rows_written": written,
                "skipped_run_dirs": skipped,
                "skipped_count": len(skipped),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
