from __future__ import annotations

import unittest

from twentyq.plot_model_overview import build_plot_rows


class PlotModelOverviewTests(unittest.TestCase):
    def test_build_plot_rows_uses_current_analysis_schema(self) -> None:
        analysis = {
            "model_summary": [
                {
                    "guesser_model": "gpt-5.4",
                    "solve_rate": 0.988,
                    "avg_turns_solved": 21.3,
                    "observed_runs": 80,
                }
            ]
        }

        rows = build_plot_rows(analysis)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].model_id, "gpt-5.4")
        self.assertEqual(rows[0].label, "GPT-5.4")
        self.assertAlmostEqual(rows[0].solve_rate, 0.988)
        self.assertAlmostEqual(rows[0].avg_turns_solved, 21.3)
        self.assertEqual(rows[0].observed_runs, 80)

    def test_build_plot_rows_skips_models_without_solved_turn_average(self) -> None:
        analysis = {
            "model_summary": [
                {
                    "guesser_model": "gpt-5.4",
                    "solve_rate": 0.988,
                    "avg_turns_solved": None,
                    "observed_runs": 80,
                },
                {
                    "guesser_model": "gpt-5.4-mini",
                    "solve_rate": 0.862,
                    "avg_turns_solved": 20.2,
                    "observed_runs": 80,
                },
            ]
        }

        rows = build_plot_rows(analysis)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].model_id, "gpt-5.4-mini")

    def test_build_plot_rows_requires_model_summary(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_summary"):
            build_plot_rows({})


if __name__ == "__main__":
    unittest.main()
