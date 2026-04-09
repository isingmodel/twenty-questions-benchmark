#!/usr/bin/env python3
"""Render grouped solve@k bars from results.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_PATH = _REPO_ROOT / "results" / "results.csv"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "img" / "solve_at_cutoffs.png"
DEFAULT_CUTOFFS = (20, 40, 60)

LABEL_MAP: dict[str, str] = {
    "gpt-4o": "GPT-4o",
    "gpt-5_low": "GPT-5 (low)",
    "gpt-5.4_low": "GPT-5.4 (low)",
    "gpt-5.4_high": "GPT-5.4 (high)",
    "gpt-5.4-mini_low": "GPT-5.4 Mini (low)",
    "gpt-5.4-mini_high": "GPT-5.4 Mini (high)",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3-flash-preview_low": "Gemini 3 Flash (low)",
    "gemini-3-flash-preview_high": "Gemini 3 Flash (high)",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "gemini-3.1-flash-lite-preview_low": "Gemini 3.1 Flash Lite (low)",
    "gemini-3.1-flash-lite-preview_high": "Gemini 3.1 Flash Lite (high)",
    "claude-opus-4-6_budget_2048": "Claude Opus 4.6 (budget 2048)",
    "claude-opus-4-6_low": "Claude Opus 4.6 (low)",
    "claude-opus-4-6_high": "Claude Opus 4.6 (high)",
    "claude-sonnet-4-5_budget_2048": "Claude Sonnet 4.5 (budget 2048)",
    "claude-sonnet-4-5_low": "Claude Sonnet 4.5 (low)",
    "claude-sonnet-4-5_high": "Claude Sonnet 4.5 (high)",
    "claude-3-7-sonnet-20250219_low": "Claude 3.7 Sonnet (low)",
    "claude-3-7-sonnet-20250219_high": "Claude 3.7 Sonnet (high)",
}

CUTOFF_COLORS = ("#7EC8E3", "#4A90D9", "#2D6FB7")


@dataclass(frozen=True)
class SolveRateRow:
    model_id: str
    label: str
    observed_runs: int
    solve_rates: dict[int, float]


def _clean_cell(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_cutoffs(raw: str) -> tuple[int, ...]:
    cutoffs: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid cutoff value {item!r}") from exc
        if value < 1:
            raise ValueError("Cutoffs must be positive integers")
        cutoffs.append(value)
    if not cutoffs:
        raise ValueError("At least one cutoff is required")
    return tuple(sorted(set(cutoffs)))


def load_solve_rate_rows(path: Path, cutoffs: tuple[int, ...]) -> list[SolveRateRow]:
    by_model: dict[str, list[tuple[bool, int]]] = defaultdict(list)

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            turns_raw = _clean_cell(row.get("turns_used"))
            if not turns_raw:
                continue
            model_id = _clean_cell(row.get("guesser_w_effort"))
            if not model_id:
                guesser_model = _clean_cell(row.get("guesser_model"))
                reasoning_effort = _clean_cell(row.get("guesser_reasoning_effort"))
                model_id = (
                    f"{guesser_model}_{reasoning_effort}" if guesser_model and reasoning_effort else guesser_model
                )
            if not model_id:
                continue
            by_model[model_id].append((_clean_cell(row.get("solved")).lower() == "true", int(turns_raw)))

    rows: list[SolveRateRow] = []
    for model_id, runs in by_model.items():
        solve_rates = {}
        for cutoff in cutoffs:
            successes = sum(1 for solved, turns in runs if solved and turns <= cutoff)
            solve_rates[cutoff] = successes / len(runs)
        rows.append(
            SolveRateRow(
                model_id=model_id,
                label=LABEL_MAP.get(model_id, model_id),
                observed_runs=len(runs),
                solve_rates=solve_rates,
            )
        )

    rows.sort(
        key=lambda row: tuple([-row.solve_rates[cutoff] for cutoff in reversed(cutoffs)] + [row.label.lower()])
    )
    if not rows:
        raise ValueError("No valid rows found in CSV")
    return rows


def render_plot(rows: list[SolveRateRow], cutoffs: tuple[int, ...], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not found, skipping plot. Please install matplotlib.")
        return

    if len(cutoffs) > len(CUTOFF_COLORS):
        raise ValueError(f"At most {len(CUTOFF_COLORS)} cutoffs are supported by the default style")

    fig_height = max(5.5, 0.68 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    group_centers = list(range(len(rows)))
    bar_height = 0.22 if len(cutoffs) == 3 else min(0.25, 0.8 / max(len(cutoffs), 1))
    offsets = [bar_height * (index - (len(cutoffs) - 1) / 2) for index in range(len(cutoffs))]

    for cutoff_index, cutoff in enumerate(cutoffs):
        positions = [center + offsets[cutoff_index] for center in group_centers]
        values = [row.solve_rates[cutoff] * 100.0 for row in rows]
        color = CUTOFF_COLORS[cutoff_index]
        bars = ax.barh(
            positions,
            values,
            height=bar_height,
            color=color,
            edgecolor="#233043",
            linewidth=0.6,
            label=f"Solve@{cutoff}",
        )
        for bar, value in zip(bars, values):
            ax.text(
                value + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#233043",
            )

    ax.set_yticks(group_centers)
    ax.set_yticklabels([row.label for row in rows], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Solve Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Twenty Questions Benchmark\nSolve@20 / Solve@40 / Solve@60", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render grouped solve@k bars from results.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to results.csv.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination image path.")
    parser.add_argument(
        "--cutoffs",
        default="20,40,60",
        help="Comma-separated solve cutoffs to render. Default: 20,40,60",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoffs = parse_cutoffs(args.cutoffs)
    rows = load_solve_rate_rows(args.input, cutoffs)
    render_plot(rows, cutoffs, args.output)

    header = " ".join(f"{f'solve@{cutoff}':>10}" for cutoff in cutoffs)
    print(f"\n{'model':<40} {header} {'n':>6}")
    print("-" * (48 + 12 * len(cutoffs)))
    for row in rows:
        rates = " ".join(f"{row.solve_rates[cutoff]:>10.2%}" for cutoff in cutoffs)
        print(f"{row.model_id:<40} {rates} {row.observed_runs:>6}")


if __name__ == "__main__":
    main()
