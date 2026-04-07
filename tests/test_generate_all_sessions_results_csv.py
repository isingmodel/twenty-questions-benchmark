from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.generate_all_sessions_results_csv import collect_all_session_rows, write_results_csv


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


class GenerateAllSessionsResultsCsvTests(unittest.TestCase):
    def test_collect_rows_uses_run_number_and_reasoning_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "all_sessions"
            openai_dir = runs_root / "run-0017__full-game-test__openai__gpt-5.4-mini"
            anthropic_dir = runs_root / "run-0003__full-game-test__anthropic__claude-opus-4-6"
            missing_dir = runs_root / "run-9999__full-game-test__openai__gpt-5.4-mini"

            _write_json(
                openai_dir / "run_config.json",
                {
                    "run_id": openai_dir.name,
                    "mode": "full-game-test",
                    "config": {
                        "target_id": "place_busan",
                        "guesser_model": "gpt-5.4-mini",
                        "judge_model": "gpt-5.4-mini",
                        "guesser_reasoning": {"reasoning_effort": "high"},
                        "judge_reasoning": {"reasoning_effort": "low"},
                    },
                },
            )
            _write_json(
                openai_dir / "summary.json",
                {
                    "run_id": openai_dir.name,
                    "mode": "full-game-test",
                    "target_id": "place_busan",
                    "target_name": "Busan",
                    "solved": True,
                    "turns_used": 14,
                    "final_question": "Is the hidden target Busan?",
                    "final_question_correct": True,
                },
            )

            _write_json(
                anthropic_dir / "run_config.json",
                {
                    "run_id": anthropic_dir.name,
                    "mode": "full-game-test",
                    "config": {
                        "target_id": "character_gandalf",
                        "guesser_model": "claude-opus-4-6",
                        "judge_model": "gpt-5.4-mini",
                        "guesser_reasoning": {"thinking_budget": 2048},
                        "judge_reasoning": {"reasoning_effort": "low"},
                    },
                },
            )
            _write_json(
                anthropic_dir / "summary.json",
                {
                    "run_id": anthropic_dir.name,
                    "mode": "full-game-test",
                    "target_id": "character_gandalf",
                    "target_name": "Gandalf",
                    "solved": False,
                    "turns_used": 80,
                    "final_question": "Is the character Gandalf?",
                    "final_question_correct": False,
                },
            )

            _write_json(
                missing_dir / "run_config.json",
                {
                    "run_id": missing_dir.name,
                    "mode": "full-game-test",
                    "config": {
                        "target_id": "food_croissant",
                        "guesser_model": "gpt-5.4-mini",
                        "judge_model": "gpt-5.4-mini",
                    },
                },
            )

            rows, skipped = collect_all_session_rows(runs_root)

        self.assertEqual(len(rows), 2)
        self.assertEqual(skipped, [missing_dir.name])

        first_row = rows[0]
        self.assertEqual(first_row["run_id"], anthropic_dir.name)
        self.assertEqual(first_row["repetition_index"], 3)
        self.assertEqual(first_row["variant_label"], "claude-opus-4-6")
        self.assertEqual(first_row["guesser_reasoning_effort"], "budget_2048")
        self.assertEqual(first_row["guesser_w_effort"], "claude-opus-4-6_budget_2048")

        second_row = rows[1]
        self.assertEqual(second_row["run_id"], openai_dir.name)
        self.assertEqual(second_row["repetition_index"], 17)
        self.assertEqual(second_row["guesser_reasoning_effort"], "high")
        self.assertEqual(second_row["judge_reasoning_effort"], "low")
        self.assertEqual(second_row["guesser_w_effort"], "gpt-5.4-mini_high")

    def test_write_results_csv_emits_expected_header_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs_root = root / "all_sessions"
            output_path = root / "results" / "results_all_sessions.csv"
            gemini_dir = runs_root / "run-0001__full-game-test__gemini__gemini-3.1-flash-lite-preview__26-04-06"

            _write_json(
                gemini_dir / "run_config.json",
                {
                    "run_id": "run-0001__full-game-test__gemini__gemini-3.1-flash-lite-preview",
                    "mode": "full-game-test",
                    "config": {
                        "target_id": "place_busan",
                        "guesser_model": "gemini-3.1-flash-lite-preview",
                        "judge_model": "gpt-5.4-mini",
                        "guesser_reasoning": {"thinking_level": "low"},
                        "judge_reasoning": {"reasoning_effort": "low"},
                    },
                },
            )
            _write_json(
                gemini_dir / "summary.json",
                {
                    "run_id": "run-0001__full-game-test__gemini__gemini-3.1-flash-lite-preview",
                    "mode": "full-game-test",
                    "target_id": "place_busan",
                    "target_name": "Busan",
                    "solved": True,
                    "turns_used": 12,
                    "final_question": "Is the hidden target Busan?",
                    "final_question_correct": True,
                },
            )

            written, skipped = write_results_csv(runs_root, output_path)

            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(written, 1)
        self.assertEqual(skipped, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "run-0001__full-game-test__gemini__gemini-3.1-flash-lite-preview")
        self.assertEqual(rows[0]["repetition_index"], "1")
        self.assertEqual(rows[0]["guesser_reasoning_effort"], "")
        self.assertEqual(rows[0]["guesser_w_effort"], "gemini-3.1-flash-lite-preview")
        self.assertEqual(rows[0]["guesser_reasoning"], "{'thinking_level': 'low'}")
