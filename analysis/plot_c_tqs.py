#!/usr/bin/env python3
"""Compute and plot the C-TQS benchmark score from run-level CSV results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_PATH = _REPO_ROOT / "results" / "results.csv"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "img" / "c_tqs_model_ranking.png"


@dataclass(frozen=True)
class RunRecord:
    target_id: str
    guesser_w_effort: str
    turns_used: int
    solved: bool


@dataclass(frozen=True)
class ModelScore:
    guesser_w_effort: str
    overall_score: float
    per_target_scores: dict[str, float]
    per_target_rmq: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the censored time-to-solve score (C-TQS) by guesser_w_effort.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Run-level results CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output PNG path.")
    return parser.parse_args()


def load_records(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"target_id", "guesser_w_effort", "turns_used", "solved"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for row in reader:
            target_id = str(row["target_id"]).strip()
            guesser_w_effort = str(row["guesser_w_effort"]).strip()
            turns_raw = str(row["turns_used"]).strip()
            solved_raw = str(row["solved"]).strip().lower()
            if not target_id or not guesser_w_effort or not turns_raw:
                continue
            try:
                turns_used = int(turns_raw)
            except ValueError:
                continue
            records.append(
                RunRecord(
                    target_id=target_id,
                    guesser_w_effort=guesser_w_effort,
                    turns_used=turns_used,
                    solved=solved_raw == "true",
                )
            )
    if not records:
        raise ValueError("No valid records found in CSV")
    return records


def km_rmq(runs: Iterable[RunRecord], tau: int) -> float:
    """Restricted Mean Questions up to tau using Kaplan-Meier handling of right-censoring."""
    turns: list[int] = []
    events: list[int] = []
    for run in runs:
        y = min(max(run.turns_used, 1), tau)
        turns.append(y)
        events.append(1 if run.solved and run.turns_used <= tau else 0)

    if not turns:
        raise ValueError("Cannot compute RMQ with empty runs")

    n = len(turns)
    survival_after_turn = [1.0] * (tau + 1)
    s = 1.0
    for k in range(1, tau + 1):
        at_risk = 0
        events_at_k = 0
        for y, d in zip(turns, events, strict=False):
            if y >= k:
                at_risk += 1
            if y == k and d == 1:
                events_at_k += 1
        if at_risk > 0 and events_at_k > 0:
            s *= 1.0 - (events_at_k / at_risk)
        survival_after_turn[k] = s

    rmq = 0.0
    for k in range(0, tau):
        rmq += survival_after_turn[k]
    return rmq


def compute_scores(records: list[RunRecord]) -> tuple[list[ModelScore], dict[str, int]]:
    by_target_model: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    by_target: dict[str, list[RunRecord]] = defaultdict(list)

    for rec in records:
        by_target_model[(rec.target_id, rec.guesser_w_effort)].append(rec)
        by_target[rec.target_id].append(rec)

    target_horizons: dict[str, int] = {}
    for target_id, runs in by_target.items():
        by_model_max: dict[str, int] = defaultdict(int)
        for run in runs:
            by_model_max[run.guesser_w_effort] = max(by_model_max[run.guesser_w_effort], run.turns_used)
        target_horizons[target_id] = min(by_model_max.values())

    rmq_by_target_model: dict[tuple[str, str], float] = {}
    models = sorted(set(r.guesser_w_effort for r in records))
    targets = sorted(by_target.keys())
    for target_id in targets:
        tau = target_horizons[target_id]
        for model in models:
            runs = by_target_model.get((target_id, model), [])
            if not runs:
                continue
            rmq_by_target_model[(target_id, model)] = km_rmq(runs, tau)

    score_by_target_model: dict[tuple[str, str], float] = {}
    for target_id in targets:
        target_values = {
            model: value
            for (target, model), value in rmq_by_target_model.items()
            if target == target_id
        }
        if not target_values:
            continue
        best = min(target_values.values())
        worst = max(target_values.values())
        spread = worst - best
        for model, rmq in target_values.items():
            if spread <= 1e-9:
                score = 50.0
            else:
                score = 100.0 * ((worst - rmq) / spread)
            score_by_target_model[(target_id, model)] = score

    model_scores: list[ModelScore] = []
    for model in models:
        per_target_scores = {
            target: score
            for (target, m), score in score_by_target_model.items()
            if m == model
        }
        per_target_rmq = {
            target: rmq
            for (target, m), rmq in rmq_by_target_model.items()
            if m == model
        }
        if not per_target_scores:
            continue
        overall_score = sum(per_target_scores.values()) / len(per_target_scores)
        model_scores.append(
            ModelScore(
                guesser_w_effort=model,
                overall_score=overall_score,
                per_target_scores=per_target_scores,
                per_target_rmq=per_target_rmq,
            )
        )

    model_scores.sort(key=lambda row: row.overall_score, reverse=True)
    return model_scores, target_horizons


def _plot_scores_matplotlib(model_scores: list[ModelScore], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not model_scores:
        raise ValueError("No model scores to plot")

    targets = sorted({target for row in model_scores for target in row.per_target_scores})
    target_palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
    ]
    target_colors = {target: target_palette[idx % len(target_palette)] for idx, target in enumerate(targets)}

    y_labels = [row.guesser_w_effort for row in model_scores]
    y_positions = list(range(len(model_scores)))

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.9 * len(model_scores))))

    for y, row in zip(y_positions, model_scores, strict=False):
        for target, score in row.per_target_scores.items():
            ax.scatter(
                score,
                y,
                color=target_colors[target],
                s=52,
                alpha=0.75,
                linewidths=0.0,
                zorder=2,
            )
        ax.scatter(
            row.overall_score,
            y,
            color="black",
            marker="D",
            s=92,
            zorder=3,
        )

    ax.set_xlim(-2, 102)
    ax.set_xlabel("C-TQS (higher is better)", fontsize=12, fontweight="bold")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(
        "C-TQS by guesser_w_effort\n"
        "Dots: per-target normalized censored efficiency | Diamond: macro average",
        fontsize=13,
        fontweight="bold",
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=target_colors[t], markersize=7, label=t) for t in targets
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black", markersize=7, label="overall")
    )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved -> {output_path}")


def _plot_scores_svg(model_scores: list[ModelScore], output_path: Path) -> None:
    if not model_scores:
        raise ValueError("No model scores to plot")

    targets = sorted({target for row in model_scores for target in row.per_target_scores})
    target_palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
    ]
    target_colors = {target: target_palette[idx % len(target_palette)] for idx, target in enumerate(targets)}

    width = 1200
    row_h = 62
    top = 90
    left = 260
    right = 80
    bottom = 80
    height = top + bottom + row_h * len(model_scores)
    x0 = left
    x1 = width - right

    def x_map(score: float) -> float:
        return x0 + (score / 100.0) * (x1 - x0)

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        '<text x="600" y="36" text-anchor="middle" font-size="23" font-weight="700">'
        "C-TQS by guesser_w_effort"
        "</text>"
    )
    lines.append(
        '<text x="600" y="62" text-anchor="middle" font-size="14" fill="#444">'
        "Dots: per-target normalized censored efficiency | Diamond: macro average"
        "</text>"
    )

    for tick in range(0, 101, 10):
        x = x_map(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#e0e0e0"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-size="12" fill="#444">{tick}</text>'
        )

    lines.append(
        f'<line x1="{x0}" y1="{height-bottom}" x2="{x1}" y2="{height-bottom}" stroke="#999" stroke-width="1.2"/>'
    )
    lines.append(
        f'<text x="{(x0+x1)/2:.1f}" y="{height-24}" text-anchor="middle" font-size="14" font-weight="700">C-TQS (higher is better)</text>'
    )

    for idx, row in enumerate(model_scores):
        y = top + row_h * idx + row_h / 2
        lines.append(f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-size="14">{row.guesser_w_effort}</text>')
        for target, score in row.per_target_scores.items():
            cx = x_map(score)
            color = target_colors[target]
            lines.append(f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="6.2" fill="{color}" fill-opacity="0.78"/>')
        cx_overall = x_map(row.overall_score)
        diamond = [
            (cx_overall, y - 8.0),
            (cx_overall + 8.0, y),
            (cx_overall, y + 8.0),
            (cx_overall - 8.0, y),
        ]
        points = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in diamond)
        lines.append(f'<polygon points="{points}" fill="#111"/>')

    legend_x = x1 - 235
    legend_y = top + 10
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="225" height="{22 + 20 * len(targets)}" fill="#ffffffdd" stroke="#ddd"/>')
    for i, target in enumerate(targets):
        y = legend_y + 18 + i * 18
        lines.append(f'<circle cx="{legend_x + 12}" cy="{y - 4}" r="5" fill="{target_colors[target]}"/>')
        lines.append(f'<text x="{legend_x + 24}" y="{y}" font-size="11">{target}</text>')
    y = legend_y + 18 + len(targets) * 18
    points = f"{legend_x + 12},{y-10} {legend_x + 18},{y-4} {legend_x + 12},{y+2} {legend_x + 6},{y-4}"
    lines.append(f'<polygon points="{points}" fill="#111"/>')
    lines.append(f'<text x="{legend_x + 24}" y="{y}" font-size="11">overall</text>')

    lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved -> {output_path} (SVG fallback)")


def plot_scores(model_scores: list[ModelScore], output_path: Path) -> None:
    try:
        _plot_scores_matplotlib(model_scores, output_path)
    except ModuleNotFoundError:
        svg_output = output_path.with_suffix(".svg")
        _plot_scores_svg(model_scores, svg_output)



def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    model_scores, target_horizons = compute_scores(records)
    plot_scores(model_scores, args.output)

    print("\n=== C-TQS Summary ===")
    print("target horizons (tau):")
    for target, tau in sorted(target_horizons.items()):
        print(f"  - {target}: {tau}")

    for rank, row in enumerate(model_scores, start=1):
        print(f"{rank:>2}. {row.guesser_w_effort:<36} C-TQS={row.overall_score:6.2f}")


if __name__ == "__main__":
    main()
