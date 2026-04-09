from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analysis.plot_solve_at_cutoffs import load_solve_rate_rows, parse_cutoffs


class PlotSolveAtCutoffsTests(unittest.TestCase):
    def test_parse_cutoffs_sorts_and_deduplicates(self) -> None:
        self.assertEqual(parse_cutoffs("60,20,40,20"), (20, 40, 60))

        with self.assertRaisesRegex(ValueError, "positive"):
            parse_cutoffs("0,20")

    def test_load_solve_rate_rows_restores_missing_guesser_w_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_id",
                        "guesser_model",
                        "guesser_reasoning_effort",
                        "guesser_w_effort",
                        "turns_used",
                        "solved",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_id": "place_busan",
                        "guesser_model": "gpt-5",
                        "guesser_reasoning_effort": "low",
                        "guesser_w_effort": None,
                        "turns_used": "15",
                        "solved": "True",
                    }
                )
                writer.writerow(
                    {
                        "target_id": "place_busan",
                        "guesser_model": "gpt-5",
                        "guesser_reasoning_effort": "low",
                        "guesser_w_effort": None,
                        "turns_used": "45",
                        "solved": "True",
                    }
                )

            rows = load_solve_rate_rows(path, (20, 40, 60))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].model_id, "gpt-5_low")
        self.assertAlmostEqual(rows[0].solve_rates[20], 0.5)
        self.assertAlmostEqual(rows[0].solve_rates[40], 0.5)
        self.assertAlmostEqual(rows[0].solve_rates[60], 1.0)


if __name__ == "__main__":
    unittest.main()
