from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twentyq.run_logs import RunLogger


class RunLoggerTests(unittest.TestCase):
    def test_create_uses_highest_existing_run_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            (runs_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
            (runs_dir / "run-0002__older").mkdir()
            (runs_dir / "run-0010__latest").mkdir()

            logger = RunLogger.create(
                runs_dir=runs_dir,
                split="full-game-test",
                guesser_provider="gemini",
                guesser_model="model",
            )

        self.assertEqual(logger.run_id, "run-0011__full-game-test__gemini__model")
