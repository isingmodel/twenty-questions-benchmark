from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_targets
from .env import load_dotenv
from .episode_runner import (
    DEFAULT_BUDGET,
    DEFAULT_GUESSER_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_TARGET_ID,
    THINKING_LEVEL_CHOICES,
    FullGameConfig,
    _validate_budget,
    _validate_reasoning_config,
    run_full_game_episode,
)
from .prompts import DEFAULT_GUESSER_PROMPT_SET, ROOT


def parse_args() -> FullGameConfig:
    parser = argparse.ArgumentParser(description="Run a logged single-target Twenty Questions game.")
    parser.add_argument("--target-id", default=DEFAULT_TARGET_ID, help="Target id to use for the test game.")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Maximum number of turns before stopping.")
    parser.add_argument("--guesser-model", default=DEFAULT_GUESSER_MODEL, help="Model id for the guesser.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Model id for the judge.")
    parser.add_argument(
        "--guesser-prompt-set",
        default=DEFAULT_GUESSER_PROMPT_SET,
        help="Built-in guesser prompt set name, or a custom label when using prompt-path overrides.",
    )
    parser.add_argument(
        "--guesser-initial-prompt-path",
        type=Path,
        default=None,
        help="Optional path to a custom initial guesser prompt.",
    )
    parser.add_argument(
        "--guesser-turn-prompt-path",
        type=Path,
        default=None,
        help="Optional path to a custom per-turn guesser prompt.",
    )
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
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run directory root.")
    args = parser.parse_args()
    return FullGameConfig(
        target_id=args.target_id,
        budget=_validate_budget(args.budget),
        guesser_model=args.guesser_model,
        judge_model=args.judge_model,
        guesser_reasoning=_validate_reasoning_config(
            role="Guesser",
            model=args.guesser_model,
            thinking_level=args.guesser_thinking_level,
            thinking_budget=args.guesser_thinking_budget,
        ),
        judge_reasoning=_validate_reasoning_config(
            role="Judge",
            model=args.judge_model,
            thinking_level=args.judge_thinking_level,
            thinking_budget=args.judge_thinking_budget,
        ),
        run_dir=args.run_dir,
        guesser_prompt_set=args.guesser_prompt_set,
        guesser_initial_prompt_path=args.guesser_initial_prompt_path,
        guesser_turn_prompt_path=args.guesser_turn_prompt_path,
    )


def main() -> int:
    config = parse_args()
    load_dotenv(ROOT / ".env")
    targets = load_targets(ROOT / "data" / "all_target.csv")
    if config.target_id not in targets:
        print(f"ERROR: unknown target id {config.target_id!r}")
        return 1
    exit_code, summary = run_full_game_episode(
        config=config,
        target=targets[config.target_id],
        runs_dir=config.run_dir or ROOT / "runs",
    )
    if exit_code == 0:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    else:
        print(f"ERROR: {summary['error']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
