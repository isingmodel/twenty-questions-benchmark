#!/usr/bin/env python3
"""Generate a scatter-plot comparing guesser-model performance.

The chart plots **solve rate** (x-axis) against **average turns to solve**
(y-axis, inverted so fewer turns appears at the top).  The upper-right
corner represents the ideal model: high solve rate *and* low turn count.

Data source
-----------
Reads ``aggregate.json`` produced by :mod:`twentyq.analyze_single_target_suite`.
The file is expected at
``benchmark_analysis/claude_benchmark_analysis/aggregate.json`` relative to
the repository root.

Output
------
``img/model_overview.png`` in the repository root (created automatically if
the ``img/`` directory does not yet exist).

Usage
-----
Run as a standalone script from the repository root::

    python -m twentyq.plot_model_overview

Or call :func:`main` programmatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------
# Paths – resolved relative to the repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = _REPO_ROOT / "benchmark_analysis" / "claude_benchmark_analysis" / "aggregate.json"
"""Path to the aggregated benchmark JSON produced by the analysis pipeline."""

OUT_PATH = _REPO_ROOT / "img" / "model_overview.png"
"""Destination path for the generated chart image."""

# ---------------------------------------------------------------------------
# Visual style mappings  (model-id  →  display properties)
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, str] = {
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
}
"""Human-readable labels for each guesser model."""

COLORS: dict[str, str] = {
    "gpt-5.4": "#4A90D9",
    "gpt-5.4-mini": "#7EC8E3",
    "gemini-3-flash-preview": "#E8793A",
    "gemini-3.1-flash-lite-preview": "#F5B041",
}
"""Hex colour per model – blue family for OpenAI, orange family for Google."""

MARKERS: dict[str, str] = {
    "gpt-5.4": "o",
    "gpt-5.4-mini": "s",
    "gemini-3-flash-preview": "D",
    "gemini-3.1-flash-lite-preview": "^",
}
"""Matplotlib marker shape per model for additional visual distinction."""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load aggregate data and render the scatter-plot to *OUT_PATH*.

    Each model is placed as a single point whose position encodes both
    solve rate and turn efficiency.  An inverted y-axis ensures that
    *better* performance (fewer turns) appears toward the top of the
    chart, making the upper-right corner the "ideal" zone.

    Annotations next to each point show the exact solve rate, average
    turns, and sample size.
    """
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    model_stats: list[dict] = data["model_stats"]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Soft green shading to highlight the "ideal" region (high solve-rate,
    # low turn count).
    ax.axhspan(0, 19, xmin=0, xmax=1, color="#e8f5e9", alpha=0.4, zorder=0)
    ax.axvspan(95, 105, color="#e8f5e9", alpha=0.4, zorder=0)

    for m in model_stats:
        mid = m["guesser_model"]
        x = m["solve_rate"] * 100        # percentage
        y = m["turns_solved_mean"]       # avg turns (solved games only)
        n = m["n"]                       # number of runs
        label = LABEL_MAP.get(mid, mid)
        color = COLORS.get(mid, "#888")
        marker = MARKERS.get(mid, "o")

        ax.scatter(
            x, y,
            s=260, c=color, marker=marker,
            edgecolors="white", linewidths=1.2,
            zorder=5, label=label,
        )

        # Annotated label with key stats
        text = f"{label}\n({x:.1f}%, {y:.1f}t, n={n})"
        offset = _label_offset(mid)
        ax.annotate(
            text,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5, fontweight="bold", color=color,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, lw=0.8),
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=6,
        )

    # --- Axes & title -------------------------------------------------------
    ax.set_xlabel("Solve Rate (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Turns to Solve  (lower = better)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Twenty Questions Benchmark — Model Performance\n"
        "Solve Rate vs Turn Efficiency (All Targets)",
        fontsize=14, fontweight="bold", pad=14,
    )

    ax.set_xlim(82, 102)
    ax.set_ylim(15, 25)
    ax.invert_yaxis()  # fewer turns at the top

    # "ideal" corner hint
    ax.text(
        101, 15.5, "ideal",
        fontsize=9, fontstyle="italic", color="#4caf50", alpha=0.7,
        ha="right", va="top",
    )

    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9, markerscale=0.5)

    # --- Save ---------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_offset(model_id: str) -> tuple[int, int]:
    """Return hand-tuned ``(dx, dy)`` pixel offsets to prevent label overlap.

    The offsets are specific to the current set of four models and their
    approximate positions on the chart.  If new models are added or data
    shifts significantly, these values may need re-tuning.
    """
    return {
        "gpt-5.4": (50, 25),
        "gpt-5.4-mini": (-55, -30),
        "gemini-3-flash-preview": (55, -20),
        "gemini-3.1-flash-lite-preview": (-55, 25),
    }.get(model_id, (0, -30))


if __name__ == "__main__":
    main()
