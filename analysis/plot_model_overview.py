#!/usr/bin/env python3
"""Generate a scatter plot from cross-suite benchmark analysis output."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_PATH = _REPO_ROOT / "reports" / "single-target-suite" / "benchmark-analysis" / "aggregate.json"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "img" / "model_overview.png"

LABEL_MAP: dict[str, str] = {
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
}

COLORS: dict[str, str] = {
    "gpt-5.4": "#4A90D9",
    "gpt-5.4-mini": "#7EC8E3",
    "gemini-3-flash-preview": "#E8793A",
    "gemini-3.1-flash-lite-preview": "#F5B041",
    "claude-opus-4-6": "#2F6B5F",
    "claude-sonnet-4-5": "#53A08E",
    "claude-3-7-sonnet-20250219": "#8CC7B8",
}

MARKERS: dict[str, str] = {
    "gpt-5.4": "o",
    "gpt-5.4-mini": "s",
    "gemini-3-flash-preview": "D",
    "gemini-3.1-flash-lite-preview": "^",
    "claude-opus-4-6": "P",
    "claude-sonnet-4-5": "X",
    "claude-3-7-sonnet-20250219": "v",
}


@dataclass(frozen=True)
class PlotRow:
    model_id: str
    label: str
    solve_rate: float
    turns_per_success_at_horizon: float
    observed_runs: int
    solve_rate_stddev: float = 0.0
    turns_per_success_at_horizon_stddev: float = 0.0


def _require_float(value: Any, field_name: str, model_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric field {field_name!r} for model {model_id!r}")
    return float(value)


def _require_int(value: Any, field_name: str, model_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer field {field_name!r} for model {model_id!r}")
    return value


def build_plot_rows(analysis: dict[str, Any]) -> list[PlotRow]:
    raw_rows = analysis.get("model_summary")
    if not isinstance(raw_rows, list):
        raise ValueError("Expected list field 'model_summary' in aggregate analysis")

    rows: list[PlotRow] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        model_id = str(raw_row.get("guesser_model", "")).strip()
        if not model_id:
            continue
        turns_per_success_at_horizon = raw_row.get("turns_per_success_at_horizon")
        if turns_per_success_at_horizon is None:
            continue
        rows.append(
            PlotRow(
                model_id=model_id,
                label=LABEL_MAP.get(model_id, model_id),
                solve_rate=_require_float(raw_row.get("solve_rate"), "solve_rate", model_id),
                turns_per_success_at_horizon=_require_float(
                    turns_per_success_at_horizon,
                    "turns_per_success_at_horizon",
                    model_id,
                ),
                observed_runs=_require_int(raw_row.get("observed_runs"), "observed_runs", model_id),
                solve_rate_stddev=max(
                    0.0,
                    _require_float(raw_row.get("repetition_solve_rate_stddev", 0.0), "repetition_solve_rate_stddev", model_id),
                ),
                turns_per_success_at_horizon_stddev=max(
                    0.0,
                    _require_float(
                        raw_row.get("repetition_turns_per_success_at_horizon_stddev", 0.0) or 0.0,
                        "repetition_turns_per_success_at_horizon_stddev",
                        model_id,
                    ),
                ),
            )
        )

    if not rows:
        raise ValueError("No plottable rows found in aggregate analysis")
    return rows


def _label_offset(model_id: str, index: int) -> tuple[int, int]:
    preferred = {
        "gpt-5.4": (50, 24),
        "gpt-5.4-mini": (-55, -30),
        "gemini-3-flash-preview": (55, -20),
        "gemini-3.1-flash-lite-preview": (-55, 24),
    }
    if model_id in preferred:
        return preferred[model_id]
    return (0, 20) if index % 2 == 0 else (0, -24)


def _axis_limits(rows: list[PlotRow]) -> tuple[float, float, float, float]:
    solve_rates = [row.solve_rate * 100.0 for row in rows]
    turns_per_success = [row.turns_per_success_at_horizon for row in rows]
    solve_rate_stddevs = [row.solve_rate_stddev * 100.0 for row in rows]
    turns_per_success_stddevs = [row.turns_per_success_at_horizon_stddev for row in rows]

    x_min = max(0.0, math.floor((min(x - e for x, e in zip(solve_rates, solve_rate_stddevs, strict=False)) - 3.0) / 5.0) * 5.0)
    x_max = min(100.0, math.ceil((max(x + e for x, e in zip(solve_rates, solve_rate_stddevs, strict=False)) + 3.0) / 5.0) * 5.0)
    if x_max <= x_min:
        x_min = max(0.0, x_min - 5.0)
        x_max = min(100.0, x_max + 5.0)

    y_min = max(0.0, math.floor(min(y - e for y, e in zip(turns_per_success, turns_per_success_stddevs, strict=False)) - 1.0))
    y_max = math.ceil(max(y + e for y, e in zip(turns_per_success, turns_per_success_stddevs, strict=False)) + 1.0)
    if y_max <= y_min:
        y_min = max(0.0, y_min - 1.0)
        y_max = y_max + 1.0

    return x_min, x_max, y_min, y_max


def render_plot(rows: list[PlotRow], output_path: Path, *, analysis_turn_horizon: int | None = None) -> None:
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    x_min, x_max, y_min, y_max = _axis_limits(rows)
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.axvspan(max(x_min, x_max - 10.0), x_max, color="#e8f5e9", alpha=0.35, zorder=0)
    ax.axhspan(y_min, min(y_min + 3.0, y_max), color="#e8f5e9", alpha=0.35, zorder=0)

    for index, row in enumerate(rows):
        x = row.solve_rate * 100.0
        y = row.turns_per_success_at_horizon
        color = COLORS.get(row.model_id, "#888888")
        marker = MARKERS.get(row.model_id, "o")

        ax.scatter(
            x,
            y,
            s=260,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=row.label,
        )
        if row.solve_rate_stddev > 0.0 or row.turns_per_success_at_horizon_stddev > 0.0:
            ax.errorbar(
                x,
                y,
                xerr=row.solve_rate_stddev * 100.0 if row.solve_rate_stddev > 0.0 else None,
                yerr=row.turns_per_success_at_horizon_stddev if row.turns_per_success_at_horizon_stddev > 0.0 else None,
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                alpha=0.55,
                zorder=4,
            )

        text = f"{row.label}\n({x:.1f}%, {y:.1f} t/success, n={row.observed_runs})"
        ax.annotate(
            text,
            xy=(x, y),
            xytext=_label_offset(row.model_id, index),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, lw=0.8),
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=6,
        )

    ax.set_xlabel("Solve Rate (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Turns per Success at Shared Horizon (lower = better)", fontsize=13, fontweight="bold")
    title_suffix = (
        f"H = {analysis_turn_horizon} observed turns"
        if isinstance(analysis_turn_horizon, int) and analysis_turn_horizon > 0
        else "shared observed solve horizon"
    )
    ax.set_title(
        "Twenty Questions Benchmark\nSolve Rate vs Horizon-Capped Turns per Success ({suffix}; error bars = repetition variance)".format(
            suffix=title_suffix
        ),
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.text(
        x_max,
        y_min,
        "ideal",
        fontsize=9,
        fontstyle="italic",
        color="#4caf50",
        alpha=0.8,
        ha="right",
        va="top",
    )
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9, markerscale=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a model-overview scatter plot from aggregate suite analysis.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to aggregate.json produced by analysis.analyze_single_target_suite.",
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
    analysis = json.loads(args.input.read_text(encoding="utf-8"))
    rows = build_plot_rows(analysis)
    summary = analysis.get("summary", {})
    analysis_turn_horizon = summary.get("analysis_turn_horizon")
    render_plot(
        rows,
        args.output,
        analysis_turn_horizon=analysis_turn_horizon if isinstance(analysis_turn_horizon, int) else None,
    )


if __name__ == "__main__":
    main()
