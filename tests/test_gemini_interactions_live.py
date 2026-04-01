from __future__ import annotations

import os
import unittest

from twentyq.clients import GeminiClient, GeminiInteractionSession
from twentyq.env import load_dotenv
from twentyq.prompts import ROOT
from twentyq.reasoning import ReasoningConfig


@unittest.skipUnless(
    os.environ.get("RUN_GEMINI_LIVE_TESTS") == "1",
    "Set RUN_GEMINI_LIVE_TESTS=1 to run Gemini live integration tests.",
)
class GeminiInteractionsLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        load_dotenv(ROOT / ".env")
        api_key = os.environ.get("gemini_key")
        if not api_key:
            self.skipTest("Missing gemini_key in environment or .env")
        self.client = GeminiClient(api_key)

    def test_gemini_31_pro_preview_preserves_server_side_state(self) -> None:
        secret = "cobalt-otter-17"
        session = GeminiInteractionSession(
            client=self.client,
            model="gemini-3.1-pro-preview",
            system_prompt="Reply briefly and follow exact output instructions.",
            initial_user_prompt=(
                f"Remember this exact secret for later: {secret}. "
                "Do not explain it unless explicitly asked."
            ),
        )

        first = session.generate_turn("Acknowledge with READY only.", reasoning=ReasoningConfig(thinking_level="low"))
        second = session.generate_turn(
            "What exact secret did I ask you to remember? Reply with the secret only.",
            reasoning=ReasoningConfig(thinking_level="low"),
        )

        self.assertEqual(first[3], None)
        self.assertTrue(first[2].startswith("interaction-") or len(first[2]) > 8)
        self.assertEqual(second[3], first[2])
        self.assertEqual(second[1], "What exact secret did I ask you to remember? Reply with the secret only.")
        self.assertIn(secret, second[0])


if __name__ == "__main__":
    unittest.main()
