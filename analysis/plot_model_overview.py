#!/usr/bin/env python3
"""Generate a scatter plot from results.csv, grouping by guesser_w_effort."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_PATH = _REPO_ROOT / "results" / "results.csv"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "img" / "model_overview.png"

# Keys are raw guesser_w_effort values from results.csv.
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

# Unique color + marker per guesser_w_effort.
_STYLE: dict[str, tuple[str, str]] = {
    # (color, marker)
    "gpt-4o":                              ("#5B6C8F", "X"),
    "gpt-5_low":                           ("#2D6FB7", "P"),
    "gpt-5.4_low":                          ("#4A90D9", "o"),
    "gpt-5.4_high":                         ("#4A90D9", "*"),
    "gpt-5.4-mini_low":                     ("#7EC8E3", "s"),
    "gpt-5.4-mini_high":                    ("#7EC8E3", "^"),
    "gemini-3-flash-preview":               ("#E8793A", "D"),
    "gemini-3-flash-preview_low":           ("#E8793A", "D"),
    "gemini-3-flash-preview_high":          ("#E8793A", "p"),
    "gemini-3.1-flash-lite-preview":        ("#F5B041", "v"),
    "gemini-3.1-flash-lite-preview_low":    ("#F5B041", "v"),
    "gemini-3.1-flash-lite-preview_high":   ("#F5B041", "h"),
    "claude-opus-4-6_budget_2048":          ("#2F6B5F", "H"),
    "claude-opus-4-6_low":                  ("#2F6B5F", "P"),
    "claude-opus-4-6_high":                 ("#2F6B5F", "H"),
    "claude-sonnet-4-5_budget_2048":        ("#53A08E", "d"),
    "claude-sonnet-4-5_low":                ("#53A08E", "X"),
    "claude-sonnet-4-5_high":               ("#53A08E", "d"),
    "claude-3-7-sonnet-20250219_low":       ("#8CC7B8", ">"),
    "claude-3-7-sonnet-20250219_high":      ("#8CC7B8", "<"),
}

_FALLBACK_COLOR = "#888888"
_FALLBACK_MARKER = "o"


def _color_for(guesser_w_effort: str) -> str:
    style = _STYLE.get(guesser_w_effort)
    return style[0] if style else _FALLBACK_COLOR


def _marker_for(guesser_w_effort: str) -> str:
    style = _STYLE.get(guesser_w_effort)
    return style[1] if style else _FALLBACK_MARKER


@dataclass(frozen=True)
class PlotRow:
    model_id: str        # guesser_w_effort
    label: str
    solve_rate: float
    turns_per_success: float   # simple avg turns for solved runs
    observed_runs: int


def load_from_csv(path: Path) -> list[PlotRow]:
    """Build PlotRow list grouped by guesser_w_effort from results.csv."""
    by_model: dict[str, list[dict]] = defaultdict(list)

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gwe = row.get("guesser_w_effort", "").strip()
            if not gwe:
                # fall back to model + effort
                gwe = row.get("guesser_model", "").strip() + "_" + row.get("guesser_reasoning_effort", "").strip()
            if not gwe or not row.get("turns_used", "").strip():
                continue
            by_model[gwe].append({
                "solved": row.get("solved", "").strip().lower() == "true",
                "turns_used": int(row["turns_used"].strip()),
            })

    rows: list[PlotRow] = []
    for gwe, runs in by_model.items():
        solved = [r for r in runs if r["solved"]]
        solve_rate = len(solved) / len(runs) if runs else 0.0
        turns_per_success = (
            sum(r["turns_used"] for r in solved) / len(solved) if solved else float("nan")
        )
        rows.append(
            PlotRow(
                model_id=gwe,
                label=LABEL_MAP.get(gwe, gwe),
                solve_rate=solve_rate,
                turns_per_success=turns_per_success,
                observed_runs=len(runs),
            )
        )

    # Skip models with no solved runs (can't put them on this chart meaningfully)
    rows = [r for r in rows if not math.isnan(r.turns_per_success)]
    rows.sort(key=lambda r: (-r.solve_rate, r.turns_per_success))
    if not rows:
        raise ValueError("No plottable rows found in CSV")
    return rows


def _axis_limits(rows: list[PlotRow]) -> tuple[float, float, float, float]:
    solve_rates = [r.solve_rate * 100.0 for r in rows]
    tps = [r.turns_per_success for r in rows]

    x_min = max(0.0, math.floor((min(solve_rates) - 3.0) / 5.0) * 5.0)
    x_max = min(100.0, math.ceil((max(solve_rates) + 3.0) / 5.0) * 5.0)
    if x_max <= x_min:
        x_min = max(0.0, x_min - 5.0)
        x_max = min(100.0, x_max + 5.0)

    y_min = max(0.0, math.floor(min(tps) - 1.0))
    y_max = math.ceil(max(tps) + 1.0)
    if y_max <= y_min:
        y_min = max(0.0, y_min - 1.0)
        y_max = y_max + 1.0

    return x_min, x_max, y_min, y_max


def render_plot(rows: list[PlotRow], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x_min, x_max, y_min, y_max = _axis_limits(rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    unmatched_style_keys = sorted({row.model_id for row in rows if row.model_id not in _STYLE})
    if unmatched_style_keys:
        print(f"Warning: using fallback scatter style for {', '.join(unmatched_style_keys)}")

    # Highlight "ideal" quadrant (high solve rate, low turns)
    ax.axvspan(max(x_min, x_max - 10.0), x_max, color="#e8f5e9", alpha=0.35, zorder=0)
    ax.axhspan(y_min, min(y_min + 3.0, y_max), color="#e8f5e9", alpha=0.35, zorder=0)

    for row in rows:
        x = row.solve_rate * 100.0
        y = row.turns_per_success
        color = _color_for(row.model_id)
        marker = _marker_for(row.model_id)

        ax.scatter(
            x,
            y,
            s=200,
            color=color,
            marker=marker,
            edgecolors="#333333",
            linewidths=0.8,
            zorder=5,
            label=row.label,
        )

    ax.set_xlabel("Solve Rate (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Turns per Success (lower = better)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Twenty Questions Benchmark\nSolve Rate vs Avg Turns per Success",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.text(
        x_max - 0.3,
        y_min + 0.15,
        "← ideal",
        fontsize=10,
        fontstyle="italic",
        fontweight="bold",
        color="#2e7d32",
        alpha=0.9,
        ha="right",
        va="top",
    )
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.92, markerscale=0.85)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a model-overview scatter plot from results.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to results.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_from_csv(args.input)
    render_plot(rows, args.output)
    print(f"\n{'guesser_w_effort':<40} {'solve_rate':>10} {'turns/success':>14} {'n':>6}")
    print("-" * 74)
    for row in rows:
        print(f"{row.model_id:<40} {row.solve_rate:>10.2%} {row.turns_per_success:>14.2f} {row.observed_runs:>6}")


if __name__ == "__main__":
    main()
