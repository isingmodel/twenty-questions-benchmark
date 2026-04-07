#!/usr/bin/env python3
"""Compute and plot the Difficulty-Weighted Efficiency Index (DWEI)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_PATH = _REPO_ROOT / "results" / "results.csv"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "img" / "weighted_efficiency_ranking.png"

LABEL_MAP: dict[str, str] = {
    "gpt-5.4_low": "GPT-5.4 (low)",
    "gpt-5.4_high": "GPT-5.4 (high)",
    "gpt-5.4-mini_low": "GPT-5.4 Mini (low)",
    "gpt-5.4-mini_high": "GPT-5.4 Mini (high)",
    "gemini-3-flash-preview": "Gemini 3 Flash (low)",
    "gemini-3-flash-preview_low": "Gemini 3 Flash (low)",
    "gemini-3-flash-preview_high": "Gemini 3 Flash (high)",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite (low)",
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


@dataclass(frozen=True)
class RunRecord:
    target_id: str
    guesser_w_effort: str
    turns_used: int
    solved: bool


@dataclass
class ModelScore:
    guesser_w_effort: str
    overall_score: float
    per_target_effs: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Difficulty-Weighted Efficiency Index.")
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
    """Restricted Mean Questions up to tau using Kaplan-Meier."""
    turns: list[int] = []
    events: list[int] = []
    for run in runs:
        y = min(max(run.turns_used, 1), tau)
        turns.append(y)
        events.append(1 if run.solved and run.turns_used <= tau else 0)

    if not turns:
        return 0.0

    n = len(turns)
    survival_after_turn = [1.0] * (tau + 1)
    s = 1.0
    for k in range(1, tau + 1):
        at_risk = 0
        events_at_k = 0
        for y, d in zip(turns, events):
            if y >= k:
                at_risk += 1
            if y == k and d == 1:
                events_at_k += 1
        if at_risk > 0 and events_at_k > 0:
            s *= 1.0 - (events_at_k / at_risk)
        survival_after_turn[k] = s

    rmq = sum(survival_after_turn[0:tau])
    return rmq


def compute_scores(records: list[RunRecord]) -> tuple[list[ModelScore], dict[str, float]]:
    by_target_model: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    by_target: dict[str, list[RunRecord]] = defaultdict(list)

    for rec in records:
        by_target_model[(rec.target_id, rec.guesser_w_effort)].append(rec)
        by_target[rec.target_id].append(rec)

    models = sorted(list(set(r.guesser_w_effort for r in records)))
    targets = sorted(list(by_target.keys()))

    target_horizons: dict[str, int] = {}
    for target_id, runs in by_target.items():
        target_horizons[target_id] = max(r.turns_used for r in runs)
        
    rmq_by_target_model: dict[tuple[str, str], float] = {}
    for target_id in targets:
        tau = target_horizons[target_id]
        for model in models:
            runs = by_target_model.get((target_id, model), [])
            if runs:
                rmq_by_target_model[(target_id, model)] = km_rmq(runs, tau)

    difficulty_by_target: dict[str, float] = {}
    for target_id in targets:
        rmqs = [rmq_by_target_model.get((target_id, m)) for m in models if (target_id, m) in rmq_by_target_model]
        if rmqs:
            difficulty_by_target[target_id] = sum(rmqs) / len(rmqs)
            
    model_scores: list[ModelScore] = []
    for model in models:
        effs = {}
        for target_id in targets:
            rmq = rmq_by_target_model.get((target_id, model))
            diff = difficulty_by_target.get(target_id)
            if rmq is not None and diff is not None and rmq > 0:
                # E = 100 * (Difficulty / RMQ)
                # 100 is benchmark average. Higher is faster/better.
                eff = 100.0 * (diff / rmq)
                effs[target_id] = eff
                
        if effs:
            overall = sum(effs.values()) / len(effs)
            model_scores.append(ModelScore(
                guesser_w_effort=model,
                overall_score=overall,
                per_target_effs=effs
            ))
            
    # Sort descending: higher efficiency is better
    model_scores.sort(key=lambda s: s.overall_score, reverse=True)
    return model_scores, difficulty_by_target


def plot_scores_matplotlib(scores: list[ModelScore], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not found, skipping plot. Please install matplotlib.")
        return

    y_labels = [LABEL_MAP.get(row.guesser_w_effort, row.guesser_w_effort) for row in scores]
    y_positions = list(range(len(scores)))
    unmatched_label_keys = sorted({row.guesser_w_effort for row in scores if row.guesser_w_effort not in LABEL_MAP})
    if unmatched_label_keys:
        print(f"Warning: using raw labels for {', '.join(unmatched_label_keys)}")

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.85 * len(scores))))

    for y, row in zip(y_positions, scores):
        color = "#10B981" if row.overall_score >= 100 else "#EF4444"
        ax.barh(y, row.overall_score, height=0.5, color=color, alpha=0.85)
        
        # Add text label for the score
        ax.text(
            row.overall_score + 2.0,
            y,
            f"{row.overall_score:.1f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#065F46" if row.overall_score >= 100 else "#991B1B",
        )

    ax.axvline(x=100.0, color="#6B7280", linestyle="--", linewidth=1.5, zorder=0)

    ax.set_xlabel("Efficiency Index (100 = Benchmark Average, Higher is Better)", fontsize=12, fontweight="bold")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight="medium")
    
    # Invert y-axis to have the best (highest score) at the top
    ax.invert_yaxis()
    
    ax.grid(axis="x", alpha=0.3, linestyle="--", color="gray")
    ax.set_axisbelow(True)

    ax.set_title(
        "Difficulty-Weighted Efficiency Index (DWEI)\nScores > 100 mean faster than average",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved DWEI plot -> {output_path}")


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    model_scores, diff_map = compute_scores(records)
    
    print("\n" + "="*80)
    print(" DIFFICULTY-WEIGHTED EFFICIENCY INDEX (DWEI)".center(80))
    print("="*80)
    print("Baseline (100) = Average Efficiency among evaluated models")
    print("-" * 80)
    print(f"{'Rank':<5} | {'Model':<35} | {'Score (>100 is Better)'}")
    print("-" * 80)
    
    for rank, score in enumerate(model_scores, start=1):
        print(f"{rank:<5} | {score.guesser_w_effort:<35} | {score.overall_score:>6.1f} pts")
        
    print("-" * 80)
    
    print("\nTop 3 Hardest Targets (Difficulty = Avg RMQ):")
    sorted_diffs = sorted(diff_map.items(), key=lambda x: x[1], reverse=True)
    for tgt, d in sorted_diffs[:3]:
        print(f"  - {tgt}: {d:.1f} avg turns")

    print("\nTop 3 Easiest Targets:")
    for tgt, d in sorted_diffs[-3:]:
        print(f"  - {tgt}: {d:.1f} avg turns")
        
    print("="*80 + "\n")

    plot_scores_matplotlib(model_scores, args.output)


if __name__ == "__main__":
    main()
