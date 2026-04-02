from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.analyze_single_target_suite import build_cross_suite_analysis, render_cross_suite_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


class AnalyzeSingleTargetSuiteTests(unittest.TestCase):
    def test_build_cross_suite_analysis_merges_selected_suites_and_computes_repetition_variance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            targets_csv = root / "targets.csv"
            targets_csv.write_text(
                "\n".join(
                    [
                        "id,name,domain,aliases,description",
                        "place_paris,Paris,places,Paris,Capital city of France.",
                        "object_toothbrush,toothbrush,objects,toothbrush,Tool for cleaning teeth.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            suite_a = root / "suite_a"
            _write_json(
                suite_a / "manifest.json",
                {
                    "suite_name": "evaluation_v3",
                    "budget": 10,
                    "targets": ["place_paris"],
                    "variants": [
                        {
                            "label": "gpt-5.4-mini",
                            "guesser_model": "gpt-5.4-mini",
                            "judge_model": "gpt-5.4-mini",
                            "repetitions": 2,
                        }
                    ],
                },
            )
            _write_json(suite_a / "status.json", {"status": "completed", "completed_at": "2026-04-02T00:00:00+00:00"})
            _write_json(
                suite_a / "results.json",
                {
                    "results": [
                        {
                            "target_id": "place_paris",
                            "target_name": "Paris",
                            "variant_label": "gpt-5.4-mini",
                            "repetition_index": 1,
                            "guesser_model": "gpt-5.4-mini",
                            "judge_model": "gpt-5.4-mini",
                            "solved": True,
                            "turns_used": 3,
                            "final_question_correct": True,
                        },
                        {
                            "target_id": "place_paris",
                            "target_name": "Paris",
                            "variant_label": "gpt-5.4-mini",
                            "repetition_index": 2,
                            "guesser_model": "gpt-5.4-mini",
                            "judge_model": "gpt-5.4-mini",
                            "solved": False,
                            "turns_used": 5,
                            "final_question_correct": False,
                        },
                    ]
                },
            )

            suite_b = root / "suite_b"
            _write_json(
                suite_b / "manifest.json",
                {
                    "suite_name": "evaluation_v3_claude",
                    "budget": 10,
                    "targets": ["object_toothbrush"],
                    "variants": [
                        {
                            "label": "claude-sonnet-4-5",
                            "guesser_model": "claude-sonnet-4-5",
                            "judge_model": "gpt-5.4-mini",
                            "repetitions": 2,
                        }
                    ],
                },
            )
            _write_json(suite_b / "status.json", {"status": "completed", "completed_at": "2026-04-02T00:00:00+00:00"})
            _write_json(
                suite_b / "results.json",
                {
                    "results": [
                        {
                            "target_id": "object_toothbrush",
                            "target_name": "toothbrush",
                            "variant_label": "claude-sonnet-4-5",
                            "repetition_index": 1,
                            "guesser_model": "claude-sonnet-4-5",
                            "judge_model": "gpt-5.4-mini",
                            "solved": True,
                            "turns_used": 4,
                            "final_question_correct": True,
                        },
                        {
                            "target_id": "object_toothbrush",
                            "target_name": "toothbrush",
                            "variant_label": "claude-sonnet-4-5",
                            "repetition_index": 2,
                            "guesser_model": "claude-sonnet-4-5",
                            "judge_model": "gpt-5.4-mini",
                            "solved": True,
                            "turns_used": 6,
                            "final_question_correct": True,
                        },
                    ]
                },
            )

            analysis = build_cross_suite_analysis(
                reports_root=root,
                targets_dir=targets_csv,
                suite_dirs=[suite_a, suite_b],
                include_running=False,
            )

        self.assertEqual(analysis["summary"]["suite_count"], 2)
        self.assertEqual(len(analysis["selected_suite_dirs"]), 2)
        self.assertEqual(len(analysis["model_repetition_summary"]), 4)

        gpt_row = next(row for row in analysis["model_summary"] if row["guesser_model"] == "gpt-5.4-mini")
        self.assertEqual(gpt_row["repetition_bucket_count"], 2)
        self.assertEqual(gpt_row["analysis_turn_horizon"], 6)
        self.assertAlmostEqual(gpt_row["repetition_solve_rate_stddev"], 0.5)
        self.assertAlmostEqual(gpt_row["repetition_avg_turns_used_stddev"], 1.0)
        self.assertAlmostEqual(gpt_row["repetition_avg_turns_solved_stddev"], 0.0)
        self.assertAlmostEqual(gpt_row["avg_penalized_turns"], 6.5)
        self.assertAlmostEqual(gpt_row["avg_turns_capped_at_horizon"], 4.0)
        self.assertAlmostEqual(gpt_row["turns_per_success_at_horizon"], 8.0)
        self.assertAlmostEqual(gpt_row["solve_curve_auc_at_horizon"], 1 / 3)
        self.assertEqual(gpt_row["uncensored_by_horizon_runs"], 1)
        self.assertAlmostEqual(gpt_row["uncensored_by_horizon_rate"], 0.5)
        self.assertEqual(gpt_row["censored_before_horizon_runs"], 1)
        self.assertAlmostEqual(gpt_row["censored_before_horizon_rate"], 0.5)
        self.assertAlmostEqual(gpt_row["horizon_coverage_rate"], 0.5)
        self.assertAlmostEqual(gpt_row["expected_turns_to_solve"], 13.0)
        self.assertAlmostEqual(gpt_row["solve_efficiency"], 1 / 13)
        self.assertAlmostEqual(gpt_row["repetition_expected_turns_to_solve_stddev"], 0.0)
        self.assertAlmostEqual(gpt_row["repetition_turns_per_success_at_horizon_stddev"], 0.0)
        self.assertAlmostEqual(gpt_row["repetition_solve_curve_auc_at_horizon_stddev"], 1 / 3)

    def test_render_cross_suite_report_includes_repetition_stability_section(self) -> None:
        analysis = {
            "generated_at": "2026-04-02T00:00:00+00:00",
            "reports_root": "/tmp/reports",
            "selected_suite_dirs": ["/tmp/reports/suite_a", "/tmp/reports/suite_b"],
            "summary": {
                "analysis_turn_horizon": 6,
                "suite_count": 2,
                "completed_suite_count": 2,
                "incomplete_suite_count": 0,
                "observed_runs": 4,
                "expected_runs": 4,
                "coverage_rate": 1.0,
                "guesser_model_count": 2,
                "target_count": 2,
                "budget_mix": [{"value": "10", "count": 2, "share": 1.0}],
                "judge_model_mix": [{"value": "gpt-5.4-mini", "count": 4, "share": 1.0}],
            },
            "key_findings": [],
            "caveats": [],
            "suites": [],
            "model_summary": [
                {
                    "guesser_model": "gpt-5.4-mini",
                    "observed_runs": 2,
                    "expected_runs": 2,
                    "solve_rate": 0.5,
                    "solve_curve_auc_at_horizon": 1 / 3,
                    "avg_turns_used": 4.0,
                    "avg_penalized_turns": 6.5,
                    "analysis_turn_horizon": 6,
                    "avg_turns_capped_at_horizon": 4.0,
                    "turns_per_success_at_horizon": 8.0,
                    "uncensored_by_horizon_runs": 1,
                    "uncensored_by_horizon_rate": 0.5,
                    "censored_before_horizon_runs": 1,
                    "censored_before_horizon_rate": 0.5,
                    "horizon_covered_runs": 1,
                    "horizon_coverage_rate": 0.5,
                    "expected_turns_to_solve": 13.0,
                    "avg_turns_solved": 3.0,
                    "budget_exhaustion_rate": 0.0,
                    "error_runs": 0,
                    "targets_covered": ["place_paris"],
                    "repetition_bucket_count": 2,
                    "repetition_solve_rate_stddev": 0.5,
                    "repetition_solve_curve_auc_at_horizon_stddev": 1 / 3,
                    "repetition_turns_per_success_at_horizon_stddev": 0.0,
                    "repetition_avg_turns_used_stddev": 1.0,
                    "repetition_expected_turns_to_solve_stddev": 0.0,
                    "repetition_avg_turns_solved_stddev": 0.0,
                    "repetition_solve_rate_min": 0.0,
                    "repetition_solve_rate_max": 1.0,
                }
            ],
            "target_summary": [],
            "model_target_summary": [],
            "model_repetition_summary": [
                {
                    "repetition_index": 1,
                    "guesser_model": "gpt-5.4-mini",
                    "observed_runs": 1,
                    "expected_runs": 1,
                    "solve_rate": 1.0,
                    "solve_curve_auc_at_horizon": 2 / 3,
                    "turns_per_success_at_horizon": 3.0,
                    "censored_before_horizon_rate": 0.0,
                    "avg_turns_used": 3.0,
                    "avg_turns_solved": 3.0,
                    "budget_exhaustion_rate": 0.0,
                    "error_runs": 0,
                }
            ],
        }

        report = render_cross_suite_report(analysis)

        self.assertIn("## Repetition Stability", report)
        self.assertIn("## Metric Notes", report)
        self.assertIn("## Model x Repetition", report)
        self.assertIn("turns_per_success_h", report)
        self.assertIn("solve_curve_auc", report)
        self.assertIn("censored_before_h", report)
        self.assertIn("/tmp/reports/suite_a", report)


if __name__ == "__main__":
    unittest.main()
