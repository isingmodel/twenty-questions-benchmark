from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twentyq.run_single_target_suite import (
    ModelVariant,
    SingleTargetSuiteConfig,
    _build_jobs,
    _default_suite_dir,
    _prepare_resume_plan,
    _result_record,
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

    def test_default_suite_dir_omits_target_ids_from_folder_name(self) -> None:
        config = SingleTargetSuiteConfig(
            suite_name="evaluation_v3_claude",
            target_ids=("place_busan", "animal_octopus"),
            budget=80,
            output_dir=None,
            variants=(),
        )

        path = _default_suite_dir(config)

        self.assertIn("__evaluation_v3_claude__budget80", path.name)
        self.assertNotIn("place_busan", path.name)
        self.assertNotIn("animal_octopus", path.name)

    def test_default_suite_dir_is_unique_even_with_same_config(self) -> None:
        config = SingleTargetSuiteConfig(
            suite_name="evaluation_v3_claude",
            target_ids=("place_busan", "animal_octopus"),
            budget=80,
            output_dir=None,
            variants=(),
        )

        first = _default_suite_dir(config)
        second = _default_suite_dir(config)

        self.assertNotEqual(first.name, second.name)
        self.assertIn("__evaluation_v3_claude__budget80__", first.name)
        self.assertIn("__evaluation_v3_claude__budget80__", second.name)

    def test_prepare_resume_plan_recovers_completed_and_restarts_partial_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_dir = Path(tmpdir) / "suite"
            runs_dir = suite_dir / "runs"
            runs_dir.mkdir(parents=True)

            variant = ModelVariant(
                label="gemini-3.1-flash-lite",
                guesser_model="gemini-3.1-flash-lite-preview",
                guesser_reasoning_effort=None,
                judge_model="gpt-5.4-mini",
                judge_reasoning_effort=None,
                repetitions=3,
            )
            config = SingleTargetSuiteConfig(
                suite_name="resume-suite",
                target_ids=("place_paris",),
                budget=80,
                output_dir=None,
                variants=(variant,),
            )
            jobs = _build_jobs(config, {"place_paris": {"id": "place_paris"}})

            first_result = _result_record(
                target_id="place_paris",
                variant=variant,
                repetition_index=1,
                guesser_reasoning={},
                judge_reasoning={},
                summary={
                    "run_id": "run-0001__full-game-test__gemini__gemini-3.1-flash-lite-preview",
                    "mode": "full-game-test",
                    "target_id": "place_paris",
                    "target_name": "Paris",
                    "solved": True,
                    "turns_used": 7,
                    "final_question": "Is it Paris?",
                    "final_question_correct": True,
                    "run_dir": str(runs_dir / "run-0001__full-game-test__gemini__gemini-3.1-flash-lite-preview"),
                },
            )
            (suite_dir / "results.json").write_text(json.dumps({"results": [first_result]}), encoding="utf-8")

            completed_run_dir = runs_dir / "run-0002__full-game-test__gemini__gemini-3.1-flash-lite-preview"
            completed_run_dir.mkdir()
            (completed_run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": completed_run_dir.name,
                        "mode": "full-game-test",
                        "config": {
                            "target_id": "place_paris",
                            "budget": 80,
                            "guesser_provider": "gemini",
                            "guesser_model": variant.guesser_model,
                            "guesser_reasoning": {},
                            "judge_provider": "openai",
                            "judge_model": variant.judge_model,
                            "judge_reasoning": {},
                            "run_dir": str(completed_run_dir),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (completed_run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": completed_run_dir.name,
                        "mode": "full-game-test",
                        "target_id": "place_paris",
                        "target_name": "Paris",
                        "solved": False,
                        "turns_used": 80,
                        "final_question": "Is it in France?",
                        "final_question_correct": False,
                        "run_dir": str(completed_run_dir),
                    }
                ),
                encoding="utf-8",
            )

            partial_run_dir = runs_dir / "run-0003__full-game-test__gemini__gemini-3.1-flash-lite-preview"
            partial_run_dir.mkdir()
            (partial_run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": partial_run_dir.name,
                        "mode": "full-game-test",
                        "config": {
                            "target_id": "place_paris",
                            "budget": 80,
                            "guesser_provider": "gemini",
                            "guesser_model": variant.guesser_model,
                            "guesser_reasoning": {},
                            "judge_provider": "openai",
                            "judge_model": variant.judge_model,
                            "judge_reasoning": {},
                            "run_dir": str(partial_run_dir),
                        },
                    }
                ),
                encoding="utf-8",
            )

            plan = _prepare_resume_plan(config=config, jobs=jobs, suite_dir=suite_dir)

            self.assertEqual(plan.recovered_run_ids, [completed_run_dir.name])
            self.assertEqual(plan.deleted_partial_run_ids, [partial_run_dir.name])
            self.assertFalse(partial_run_dir.exists())
            self.assertEqual(
                [(job["target_id"], job["repetition_index"]) for job in plan.remaining_jobs],
                [("place_paris", 3)],
            )
            recovered_results = [plan.pending_results[index] for index in sorted(plan.pending_results)]
            self.assertEqual(len(recovered_results), 2)
            self.assertEqual(
                {(result["target_id"], result["repetition_index"]) for result in recovered_results},
                {("place_paris", 1), ("place_paris", 2)},
            )


if __name__ == "__main__":
    unittest.main()
