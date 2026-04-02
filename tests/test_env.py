from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from twentyq.env import get_required_env_any


class EnvTests(unittest.TestCase):
    def test_get_required_env_any_returns_first_present_name(self) -> None:
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "claude-secret"}, clear=True):
            self.assertEqual(get_required_env_any("CLAUDE_API_KEY", "ANTHROPIC_API_KEY"), "claude-secret")

    def test_get_required_env_any_raises_when_all_names_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CLAUDE_API_KEY, ANTHROPIC_API_KEY"):
                get_required_env_any("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")


if __name__ == "__main__":
    unittest.main()
