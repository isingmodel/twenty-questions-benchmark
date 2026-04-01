from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twentyq.run_single_target_suite import (
    ModelVariant,
    SingleTargetSuiteConfig,
    aggregate_results,
    load_suite_config,
    render_report,
)


class SingleTargetSuiteTests(unittest.TestCase):
    def test_load_suite_config_uses_defaults_for_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "suite.json"
            config_path.write_text(
                json.dumps(
                    {
                        "suite_name": "place-paris",
                        "targets": ["place_paris"],
                        "budget": 80,
                        "repetitions": 10,
                        "judge_model": "gemini-3-flash-preview",
                        "guesser_reasoning_effort": "medium",
                        "judge_reasoning_effort": "medium",
                        "variants": [
                            {
                                "label": "gemini-2.5-flash",
                                "guesser_model": "gemini-2.5-flash",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_suite_config(config_path)

        self.assertEqual(config.suite_name, "place-paris")
        self.assertEqual(config.target_ids, ("place_paris",))
        self.assertEqual(config.variants[0].judge_model, "gemini-3-flash-preview")
        self.assertEqual(config.variants[0].guesser_reasoning_effort, "medium")
        self.assertEqual(config.variants[0].repetitions, 10)

    def test_aggregate_results_groups_by_target_and_variant(self) -> None:
        config = SingleTargetSuiteConfig(
            suite_name="suite",
            target_ids=("place_paris",),
            budget=80,
            output_dir=None,
            variants=(
                ModelVariant(
                    label="gemini-2.5-flash",
                    guesser_model="gemini-2.5-flash",
                    guesser_reasoning_effort="medium",
                    judge_model="gemini-3-flash-preview",
                    judge_reasoning_effort="medium",
                    repetitions=2,
                ),
            ),
        )
        results = [
            {
                "target_id": "place_paris",
                "variant_label": "gemini-2.5-flash",
                "guesser_model": "gemini-2.5-flash",
                "judge_model": "gemini-3-flash-preview",
                "guesser_reasoning_effort": "medium",
                "judge_reasoning_effort": "medium",
                "guesser_reasoning": {"thinking_budget": 8192},
                "judge_reasoning": {"thinking_level": "medium"},
                "solved": True,
                "turns_used": 9,
            },
            {
                "target_id": "place_paris",
                "variant_label": "gemini-2.5-flash",
                "guesser_model": "gemini-2.5-flash",
                "judge_model": "gemini-3-flash-preview",
                "guesser_reasoning_effort": "medium",
                "judge_reasoning_effort": "medium",
                "guesser_reasoning": {"thinking_budget": 8192},
                "judge_reasoning": {"thinking_level": "medium"},
                "solved": False,
                "turns_used": 80,
            },
        ]

        aggregate = aggregate_results(config, results, Path("reports/test-suite"))

        self.assertEqual(aggregate["runs_total"], 2)
        self.assertEqual(aggregate["runs_solved"], 1)
        self.assertAlmostEqual(aggregate["solve_rate"], 0.5)
        self.assertEqual(len(aggregate["groups"]), 1)
        self.assertEqual(aggregate["groups"][0]["avg_turns_used"], 44.5)

    def test_render_report_includes_variant_rows(self) -> None:
        config = SingleTargetSuiteConfig(
            suite_name="suite",
            target_ids=("place_paris",),
            budget=80,
            output_dir=None,
            variants=(),
        )
        aggregate = {
            "completed_at": "2026-03-31T00:00:00+00:00",
            "runs_total": 2,
            "runs_solved": 1,
            "runs_failed": 0,
            "solve_rate": 0.5,
            "groups": [
                {
                    "target_id": "place_paris",
                    "variant_label": "gemini-3.0-flash",
                    "guesser_model": "gemini-3-flash-preview",
                    "judge_model": "gemini-3-flash-preview",
                    "runs_total": 2,
                    "runs_solved": 1,
                    "solve_rate": 0.5,
                    "avg_turns_used": 12.5,
                    "median_turns_used": 12.5,
                    "avg_turns_solved": 10.0,
                    "runs_failed": 0,
                    "guesser_reasoning": {"thinking_level": "medium"},
                    "judge_reasoning": {"thinking_level": "medium"},
                }
            ],
        }

        report = render_report(config, aggregate)

        self.assertIn("# Single-Target Suite Report: suite", report)
        self.assertIn("| place_paris | gemini-3.0-flash | gemini-3-flash-preview |", report)


if __name__ == "__main__":
    unittest.main()
