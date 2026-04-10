from __future__ import annotations

import unittest

from twentyq.run_benchmark import BenchmarkConfig, _default_benchmark_dir


class BenchmarkDirTests(unittest.TestCase):
    def test_default_benchmark_dir_is_unique_even_with_same_config(self) -> None:
        config = BenchmarkConfig(
            budget=80,
            guesser_model="gpt-5.4",
            judge_model="gpt-5.4-mini",
            guesser_prompt_set="default",
            guesser_initial_prompt_path=None,
            guesser_turn_prompt_path=None,
            guesser_thinking_level=None,
            judge_thinking_level=None,
            guesser_thinking_budget=None,
            judge_thinking_budget=None,
            benchmark_dir=None,
        )

        first = _default_benchmark_dir(config)
        second = _default_benchmark_dir(config)

        self.assertNotEqual(first.name, second.name)
        self.assertIn("__openai__budget80__gpt-5.4__", first.name)
        self.assertIn("__openai__budget80__gpt-5.4__", second.name)


if __name__ == "__main__":
    unittest.main()
