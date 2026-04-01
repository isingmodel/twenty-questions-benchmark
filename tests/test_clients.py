from __future__ import annotations

import unittest

from twentyq.clients import GeminiInteractionSession, OpenAIClient, OpenAIResponsesSession
from twentyq.reasoning import ReasoningConfig


class FakeGeminiClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_interaction(
        self,
        model: str,
        user_input: str,
        system_instruction: str = "",
        previous_interaction_id: str | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, object]:
        interaction_id = f"interaction-{len(self.calls) + 1}"
        self.calls.append(
            {
                "model": model,
                "user_input": user_input,
                "system_instruction": system_instruction,
                "previous_interaction_id": previous_interaction_id,
                "thinking_level": thinking_level,
                "thinking_budget": thinking_budget,
            }
        )
        return {"id": interaction_id, "outputs": [{"type": "text", "text": f"response-{len(self.calls)}"}]}

    def extract_interaction_text(self, interaction: dict[str, object]) -> str:
        outputs = interaction["outputs"]
        assert isinstance(outputs, list)
        last = outputs[-1]
        assert isinstance(last, dict)
        text = last["text"]
        assert isinstance(text, str)
        return text


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_response(
        self,
        model: str,
        input_text: str,
        instructions: str = "",
        previous_response_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, object]:
        response_id = f"resp_{len(self.calls) + 1}"
        self.calls.append(
            {
                "model": model,
                "input_text": input_text,
                "instructions": instructions,
                "previous_response_id": previous_response_id,
                "reasoning_effort": reasoning_effort,
            }
        )
        return {
            "id": response_id,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": f"response-{len(self.calls)}"}]}],
        }

    def extract_response_text(self, response: dict[str, object]) -> str:
        output = response["output"]
        assert isinstance(output, list)
        first = output[0]
        assert isinstance(first, dict)
        content = first["content"]
        assert isinstance(content, list)
        part = content[0]
        assert isinstance(part, dict)
        text = part["text"]
        assert isinstance(text, str)
        return text


class GeminiInteractionSessionTests(unittest.TestCase):
    def test_generate_turn_uses_previous_interaction_id_without_replaying_history(self) -> None:
        client = FakeGeminiClient()
        session = GeminiInteractionSession(
            client=client,
            model="gemini-2.5-flash",
            system_prompt="system",
            initial_user_prompt="initial",
        )

        first = session.generate_turn("turn one", reasoning=ReasoningConfig(thinking_budget=128))
        second = session.generate_turn("turn two", reasoning=ReasoningConfig(thinking_budget=128))

        self.assertEqual(first, ("response-1", "initial\n\nturn one", "interaction-1", None))
        self.assertEqual(second, ("response-2", "turn two", "interaction-2", "interaction-1"))
        self.assertEqual(client.calls[0]["previous_interaction_id"], None)
        self.assertEqual(client.calls[1]["previous_interaction_id"], "interaction-1")
        self.assertEqual(client.calls[1]["user_input"], "turn two")

    def test_generate_turn_uses_reasoning_config_for_shared_call_path(self) -> None:
        client = FakeGeminiClient()
        session = GeminiInteractionSession(
            client=client,
            model="gemini-2.5-flash",
            system_prompt="system",
            initial_user_prompt="initial",
        )

        result = session.generate_turn("turn one", reasoning=ReasoningConfig(reasoning_effort="low"))

        self.assertEqual(result, ("response-1", "initial\n\nturn one", "interaction-1", None))
        self.assertEqual(client.calls[0]["thinking_level"], None)
        self.assertEqual(client.calls[0]["thinking_budget"], None)


class OpenAIResponsesSessionTests(unittest.TestCase):
    def test_generate_turn_uses_previous_response_id_without_replaying_history(self) -> None:
        client = FakeOpenAIClient()
        session = OpenAIResponsesSession(
            client=client,
            model="gpt-5",
            system_prompt="system",
            initial_user_prompt="initial",
        )

        first = session.generate_turn("turn one")
        second = session.generate_turn("turn two")

        self.assertEqual(first, ("response-1", "initial\n\nturn one", "resp_1", None))
        self.assertEqual(second, ("response-2", "turn two", "resp_2", "resp_1"))
        self.assertEqual(client.calls[0]["previous_response_id"], None)
        self.assertEqual(client.calls[1]["previous_response_id"], "resp_1")
        self.assertEqual(client.calls[1]["input_text"], "turn two")

    def test_generate_turn_passes_reasoning_effort(self) -> None:
        client = FakeOpenAIClient()
        session = OpenAIResponsesSession(
            client=client,
            model="gpt-5",
            system_prompt="system",
            initial_user_prompt="initial",
        )

        session.generate_turn("turn one", reasoning=ReasoningConfig(reasoning_effort="low"))

        self.assertEqual(client.calls[0]["previous_response_id"], None)
        self.assertEqual(client.calls[0].get("reasoning_effort"), "low")


class OpenAIClientTests(unittest.TestCase):
    def test_generate_content_uses_stateless_response_call(self) -> None:
        client = OpenAIClient(api_key="test-key")
        calls: list[dict[str, object]] = []

        def fake_create_response(
            *,
            model: str,
            input_text: str,
            instructions: str = "",
            previous_response_id: str | None = None,
            reasoning_effort: str | None = None,
        ) -> dict[str, object]:
            calls.append(
                {
                    "model": model,
                    "input_text": input_text,
                    "instructions": instructions,
                    "previous_response_id": previous_response_id,
                    "reasoning_effort": reasoning_effort,
                }
            )
            return {
                "id": "resp_1",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Yes"}]}],
            }

        client.create_response = fake_create_response  # type: ignore[method-assign]

        text = client.generate_content(
            model="gpt-5.4-mini",
            system_prompt="Return a judgment.",
            user_prompt="Question: Is it in Europe?",
            reasoning=ReasoningConfig(reasoning_effort="low"),
        )

        self.assertEqual(text, "Yes")
        self.assertEqual(
            calls,
            [
                {
                    "model": "gpt-5.4-mini",
                    "input_text": "Question: Is it in Europe?",
                    "instructions": "Return a judgment.",
                    "previous_response_id": None,
                    "reasoning_effort": "low",
                }
            ],
        )
