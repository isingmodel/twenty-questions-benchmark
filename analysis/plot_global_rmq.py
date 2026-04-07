#!/usr/bin/env python3
"""Compute and plot the Global RMQ (Expected Questions to Solve) from run-level CSV results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_PATH = _REPO_ROOT / "results" / "results.csv"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "img" / "global_rmq_model_ranking.png"


@dataclass(frozen=True)
class RunRecord:
    target_id: str
    guesser_w_effort: str
    turns_used: int
    solved: bool


@dataclass
class ModelScore:
    guesser_w_effort: str
    global_rmq: float
    win_rate: float
    median_success_turns: float | None
    total_games: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Global Expected Questions to Solve (Global RMQ).")
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


def compute_global_rmq(turns: list[int], events: list[int], tau: int) -> float:
    """Area under the Kaplan-Meier survival curve up to tau (Restricted Mean Survival Time)."""
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

    # Integrate: sum of survival probabilities from 0 to tau - 1
    # This represents the expected turn count to solve
    rmq = sum(survival_after_turn[0:tau])
    return rmq


def compute_scores(records: list[RunRecord]) -> tuple[list[ModelScore], int]:
    by_model: dict[str, list[RunRecord]] = defaultdict(list)
    global_tau = 0
    
    for rec in records:
        by_model[rec.guesser_w_effort].append(rec)
        if rec.turns_used > global_tau:
            global_tau = rec.turns_used

    scores: list[ModelScore] = []
    
    for model, runs in by_model.items():
        turns = []
        events = []
        successful_turns = []
        
        for r in runs:
            # Constrain y up to tau (though records shouldn't exceed global_tau)
            y = min(max(r.turns_used, 1), global_tau)
            turns.append(y)
            # Event is 1 if solved within tau limit, else 0
            is_event = 1 if (r.solved and y <= global_tau) else 0
            events.append(is_event)
            
            if r.solved:
                successful_turns.append(y)
                
        rmq = compute_global_rmq(turns, events, global_tau)
        win_rate = (len(successful_turns) / len(runs)) * 100.0 if runs else 0.0
        median_turns = statistics.median(successful_turns) if successful_turns else None
        
        scores.append(ModelScore(
            guesser_w_effort=model,
            global_rmq=rmq,
            win_rate=win_rate,
            median_success_turns=median_turns,
            total_games=len(runs)
        ))

    # Sort ascending: lower RMQ means they solve it in fewer questions (better)
    scores.sort(key=lambda s: s.global_rmq)
    return scores, global_tau


def plot_scores_matplotlib(scores: list[ModelScore], global_tau: int, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not found, skipping plot. Please install matplotlib.")
        return

    LABEL_MAP: dict[str, str] = {
        "gpt-5.4_low": "GPT-5.4 (low)",
        "gpt-5.4_high": "GPT-5.4 (high)",
        "gpt-5.4-mini_low": "GPT-5.4 Mini (low)",
        "gpt-5.4-mini_high": "GPT-5.4 Mini (high)",
        "gemini-3-flash-preview_low": "Gemini 3 Flash (low)",
        "gemini-3-flash-preview_high": "Gemini 3 Flash (high)",
        "gemini-3.1-flash-lite-preview_low": "Gemini 3.1 Flash Lite (low)",
        "gemini-3.1-flash-lite-preview_high": "Gemini 3.1 Flash Lite (high)",
        "claude-opus-4-6_low": "Claude Opus 4.6 (low)",
        "claude-opus-4-6_high": "Claude Opus 4.6 (high)",
        "claude-sonnet-4-5_low": "Claude Sonnet 4.5 (low)",
        "claude-sonnet-4-5_high": "Claude Sonnet 4.5 (high)",
        "claude-3-7-sonnet-20250219_low": "Claude 3.7 Sonnet (low)",
        "claude-3-7-sonnet-20250219_high": "Claude 3.7 Sonnet (high)",
    }

    y_labels = [LABEL_MAP.get(row.guesser_w_effort, row.guesser_w_effort) for row in scores]
    y_positions = list(range(len(scores)))

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.85 * len(scores))))

    # Draw horizontal bars
    for y, row in zip(y_positions, scores):
        ax.barh(y, row.global_rmq, height=0.5, color="#1D4ED8", alpha=0.85)
        # Add text label for the exact RMQ value
        ax.text(
            row.global_rmq + 0.3,
            y,
            f"{row.global_rmq:.1f} turns",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1E3A8A"
        )

    ax.set_xlabel("Global Expected Questions to Solve (Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, global_tau * 1.05)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight="medium")
    
    # Invert y-axis to have the best (lowest RMQ) at the top
    ax.invert_yaxis()
    
    # Add vertical gridlines for easy reading
    ax.grid(axis="x", alpha=0.3, linestyle="--", color="gray")
    ax.set_axisbelow(True)

    ax.set_title(
        "Model Ranking by Universal Expected Turn Count\n(Area under Global Kaplan-Meier Curve)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Spine cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved Global RMQ plot -> {output_path}")


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    model_scores, global_tau = compute_scores(records)
    
    print("\n" + "="*80)
    print(" GLOBAL EXPECTED QUESTIONS TO SOLVE (GLOBAL RMQ)".center(80))
    print("="*80)
    print(f"Global Horizon (tau) derived from data: {global_tau} turns")
    print("-" * 80)
    print(f"{'Rank':<5} | {'Model':<35} | {'Exp. Turns':<11} | {'Win Rate':<10} | {'Median(Success)'}")
    print("-" * 80)
    
    for rank, score in enumerate(model_scores, start=1):
        median_str = f"{score.median_success_turns:.1f}" if score.median_success_turns is not None else "N/A"
        print(f"{rank:<5} | {score.guesser_w_effort:<35} | {score.global_rmq:<11.2f} | {score.win_rate:>5.1f}%     | {median_str:>5}")
        
    print("-" * 80)
    print("* Expected Turns (Global RMQ): Area under the Kaplan-Meier curve. LOWER IS BETTER.")
    print("* Win Rate: Simple success rate (%).")
    print("* Median(Success): Median turns for successful runs only. WARNING: Subject to survivor bias.")
    print("="*80 + "\n")

    plot_scores_matplotlib(model_scores, global_tau, args.output)


if __name__ == "__main__":
    main()
