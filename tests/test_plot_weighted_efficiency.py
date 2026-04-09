from __future__ import annotations

import unittest

from analysis.plot_weighted_efficiency import RunRecord, compute_scores, harmonic_mean


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


if __name__ == "__main__":
    unittest.main()
