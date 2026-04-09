from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analysis.plot_weighted_efficiency import RunRecord, compute_scores, harmonic_mean, load_records


class PlotWeightedEfficiencyTests(unittest.TestCase):
    def test_harmonic_mean_requires_positive_values(self) -> None:
        self.assertAlmostEqual(harmonic_mean([10.0, 20.0, 40.0]), 17.142857142857142)

        with self.assertRaisesRegex(ValueError, "positive"):
            harmonic_mean([10.0, 0.0])

    def test_compute_scores_anchors_model_average_at_100(self) -> None:
        records = [
            RunRecord(target_id="target_a", guesser_w_effort="fast", turns_used=10, solved=True),
            RunRecord(target_id="target_a", guesser_w_effort="mid", turns_used=20, solved=True),
            RunRecord(target_id="target_a", guesser_w_effort="slow", turns_used=40, solved=True),
            RunRecord(target_id="target_b", guesser_w_effort="fast", turns_used=12, solved=True),
            RunRecord(target_id="target_b", guesser_w_effort="mid", turns_used=18, solved=True),
            RunRecord(target_id="target_b", guesser_w_effort="slow", turns_used=36, solved=True),
        ]

        scores, difficulty_by_target = compute_scores(records)

        self.assertEqual(set(difficulty_by_target), {"target_a", "target_b"})
        self.assertAlmostEqual(difficulty_by_target["target_a"], 70.0 / 3.0)
        self.assertAlmostEqual(difficulty_by_target["target_b"], 22.0)

        average_score = sum(score.overall_score for score in scores) / len(scores)
        self.assertAlmostEqual(average_score, 100.0)
        self.assertEqual([score.guesser_w_effort for score in scores], ["fast", "mid", "slow"])

    def test_load_records_restores_missing_guesser_w_effort(self) -> None:
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

            records = load_records(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].guesser_w_effort, "gpt-5_low")
        self.assertEqual(records[0].turns_used, 15)
        self.assertTrue(records[0].solved)


if __name__ == "__main__":
    unittest.main()
