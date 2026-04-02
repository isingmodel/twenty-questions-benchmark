from __future__ import annotations

import unittest

from twentyq.clients import AnthropicClient, AnthropicMessagesSession, GeminiInteractionSession, OpenAIClient, OpenAIResponsesSession
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


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_message(
        self,
        model: str,
        messages: list[dict[str, object]],
        system_prompt: str = "",
        reasoning: ReasoningConfig | None = None,
        cache_system_prompt: bool = False,
    ) -> dict[str, object]:
        message_id = f"msg_{len(self.calls) + 1}"
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "system_prompt": system_prompt,
                "reasoning": reasoning,
                "cache_system_prompt": cache_system_prompt,
            }
        )
        return {
            "id": message_id,
            "content": [{"type": "text", "text": f"response-{len(self.calls)}"}],
            "usage": {
                "cache_creation_input_tokens": 120 if len(self.calls) == 1 else 0,
                "cache_read_input_tokens": 0 if len(self.calls) == 1 else 120,
                "input_tokens": 200,
                "output_tokens": 10,
            },
        }

    def extract_message_text(self, message: dict[str, object]) -> str:
        content = message["content"]
        assert isinstance(content, list)
        first = content[0]
        assert isinstance(first, dict)
        text = first["text"]
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


class AnthropicMessagesSessionTests(unittest.TestCase):
    def test_generate_turn_uses_history_with_cache_breakpoint(self) -> None:
        client = FakeAnthropicClient()
        session = AnthropicMessagesSession(
            client=client,
            model="claude-sonnet-4-20250514",
            system_prompt="system",
            initial_user_prompt="initial",
        )

        first = session.generate_turn("turn one")
        second = session.generate_turn("turn two")

        self.assertEqual(first, ("response-1", "initial\n\nturn one", "msg_1", None))
        self.assertEqual(second, ("response-2", "turn two", "msg_2", None))

        first_call_messages = client.calls[0]["messages"]
        assert isinstance(first_call_messages, list)
        self.assertEqual(len(first_call_messages), 1)
        first_call_user = first_call_messages[0]
        assert isinstance(first_call_user, dict)
        first_call_content = first_call_user["content"]
        assert isinstance(first_call_content, list)
        self.assertEqual(first_call_content[0]["text"], "initial")
        self.assertEqual(first_call_content[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(first_call_content[1]["text"], "\n\nturn one")

        second_call_messages = client.calls[1]["messages"]
        assert isinstance(second_call_messages, list)
        self.assertEqual(len(second_call_messages), 3)
        second_call_assistant = second_call_messages[1]
        assert isinstance(second_call_assistant, dict)
        second_call_assistant_content = second_call_assistant["content"]
        assert isinstance(second_call_assistant_content, list)
        self.assertEqual(second_call_assistant_content[0]["cache_control"], {"type": "ephemeral"})
        second_call_user = second_call_messages[2]
        assert isinstance(second_call_user, dict)
        second_call_user_content = second_call_user["content"]
        assert isinstance(second_call_user_content, list)
        self.assertEqual(second_call_user_content[0]["text"], "turn two")
        self.assertNotIn("cache_control", second_call_user_content[0])
        self.assertEqual(session.last_call_metadata["cache_read_input_tokens"], 120)

    def test_generate_turn_passes_thinking_budget(self) -> None:
        client = FakeAnthropicClient()
        session = AnthropicMessagesSession(
            client=client,
            model="claude-sonnet-4-20250514",
            system_prompt="system",
            initial_user_prompt="initial",
        )

        session.generate_turn("turn one", reasoning=ReasoningConfig(thinking_budget=2048))

        self.assertEqual(client.calls[0]["reasoning"], ReasoningConfig(thinking_budget=2048))


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


class AnthropicClientTests(unittest.TestCase):
    def test_generate_content_uses_cacheable_system_prompt(self) -> None:
        client = AnthropicClient(api_key="test-key")
        calls: list[dict[str, object]] = []

        def fake_create_message(
            *,
            model: str,
            messages: list[dict[str, object]],
            system_prompt: str = "",
            reasoning: ReasoningConfig | None = None,
            cache_system_prompt: bool = False,
        ) -> dict[str, object]:
            calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "reasoning": reasoning,
                    "cache_system_prompt": cache_system_prompt,
                }
            )
            return {
                "id": "msg_1",
                "content": [{"type": "text", "text": "Yes"}],
            }

        client.create_message = fake_create_message  # type: ignore[method-assign]

        text = client.generate_content(
            model="claude-sonnet-4-20250514",
            system_prompt="Return a judgment.",
            user_prompt="Question: Is it in Europe?",
            reasoning=ReasoningConfig(thinking_budget=2048),
        )

        self.assertEqual(text, "Yes")
        self.assertEqual(calls[0]["system_prompt"], "Return a judgment.")
        self.assertEqual(calls[0]["reasoning"], ReasoningConfig(thinking_budget=2048))
        self.assertEqual(calls[0]["cache_system_prompt"], True)
