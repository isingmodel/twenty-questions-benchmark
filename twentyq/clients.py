from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from .reasoning import ReasoningConfig


class HTTPError(RuntimeError):
    pass


class RetriableContentError(HTTPError):
    """API returned HTTP 200 but content is empty, blocked, or unparseable.

    These failures are often transient (safety filters, server-side load,
    thinking-only responses) and worth retrying.
    """
    pass


RETRIABLE_HTTP_CODES = {408, 429, 499, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BASE_DELAY_SECONDS = 1.5
DEFAULT_MAX_JITTER_SECONDS = 0.65
DEFAULT_PRE_REQUEST_DELAY_SECONDS = 1.5
RETRY_DELAY_SECONDS = 4.
CONTENT_RETRY_MAX_ATTEMPTS = 3
CONTENT_RETRY_DELAY_SECONDS = 3.
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 1024
ANTHROPIC_THINKING_OUTPUT_BUFFER = 1024
ANTHROPIC_EPHEMERAL_CACHE_CONTROL = {"type": "ephemeral"}


def _normalize_chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"])
        normalized.append({"role": role, "content": content})
    return normalized


def _copy_content_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for block in blocks:
        copied_block = dict(block)
        cache_control = copied_block.get("cache_control")
        if isinstance(cache_control, dict):
            copied_block["cache_control"] = dict(cache_control)
        copied.append(copied_block)
    return copied


def _should_retry_http_error(exc: urllib.error.HTTPError) -> bool:
    return exc.code in RETRIABLE_HTTP_CODES


def _compute_retry_delay(attempt_index: int) -> float:
    backoff = DEFAULT_BASE_DELAY_SECONDS * (2**attempt_index)
    jitter = random.uniform(0.0, DEFAULT_MAX_JITTER_SECONDS)
    return backoff + jitter


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt_index in range(DEFAULT_MAX_ATTEMPTS):
        time.sleep(DEFAULT_PRE_REQUEST_DELAY_SECONDS)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if attempt_index < DEFAULT_MAX_ATTEMPTS - 1 and _should_retry_http_error(exc):
                if exc.code != 429:
                    retry_delay_seconds = _compute_retry_delay(attempt_index)
                time.sleep(RETRY_DELAY_SECONDS)
                last_error = exc
                continue
            raise HTTPError(f"HTTP {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            if attempt_index < DEFAULT_MAX_ATTEMPTS - 1:
                time.sleep(_compute_retry_delay(attempt_index))
                last_error = exc
                continue
            raise HTTPError(f"Network error calling {url}: {exc.reason}") from exc
        except socket.timeout as exc:
            if attempt_index < DEFAULT_MAX_ATTEMPTS - 1:
                time.sleep(_compute_retry_delay(attempt_index))
                last_error = exc
                continue
            raise HTTPError(f"Socket timeout calling {url}") from exc
    if isinstance(last_error, urllib.error.HTTPError):
        body = last_error.read().decode("utf-8", errors="replace")
        raise HTTPError(f"HTTP {last_error.code} from {url}: {body}") from last_error
    if isinstance(last_error, urllib.error.URLError):
        raise HTTPError(f"Network error calling {url}: {last_error.reason}") from last_error
    if isinstance(last_error, socket.timeout):
        raise HTTPError(f"Socket timeout calling {url}") from last_error
    raise HTTPError(f"Unknown error calling {url}")


def _build_generate_content_config(
    thinking_level: str | None = None,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "candidateCount": 1,
    }
    thinking_config: dict[str, Any] = {}
    if thinking_level is not None:
        thinking_config["thinkingLevel"] = thinking_level
    if thinking_budget is not None:
        thinking_config["thinkingBudget"] = thinking_budget
    if thinking_config:
        config["thinkingConfig"] = thinking_config
    return config


def _build_interaction_generation_config(
    thinking_level: str | None = None,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if thinking_level is not None:
        config["thinking_level"] = thinking_level
    if thinking_budget is not None:
        config["thinking_budget"] = thinking_budget
    return config


def _with_content_retry(
    fn: Any,
    max_attempts: int = CONTENT_RETRY_MAX_ATTEMPTS,
    delay: float = CONTENT_RETRY_DELAY_SECONDS,
) -> Any:
    """Retry *fn* on RetriableContentError with back-off.

    Catches only RetriableContentError so that genuine HTTP/network
    failures (already retried inside _post_json) propagate immediately.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except RetriableContentError:
            if attempt < max_attempts - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise


def _extract_interaction_text(interaction: dict[str, Any]) -> str:
    outputs = interaction.get("outputs")
    if not isinstance(outputs, list):
        raise RetriableContentError(f"Unexpected Gemini interaction response: {json.dumps(interaction)}")
    for output in reversed(outputs):
        if not isinstance(output, dict):
            continue
        text = output.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    raise RetriableContentError(f"Gemini interaction did not return text output: {json.dumps(interaction)}")


def _extract_openai_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = response.get("output")
    if not isinstance(outputs, list):
        raise RetriableContentError(f"Unexpected OpenAI response: {json.dumps(response)}")

    chunks: list[str] = []
    for item in outputs:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"output_text", "text"}:
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    if chunks:
        return "\n".join(chunks).strip()
    raise RetriableContentError(f"OpenAI response did not return text output: {json.dumps(response)}")


def _extract_anthropic_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        raise RetriableContentError(f"Unexpected Anthropic response: {json.dumps(message)}")

    chunks: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    if chunks:
        return "\n".join(chunks).strip()
    raise RetriableContentError(f"Anthropic response did not return text output: {json.dumps(message)}")


def _thinking_kwargs_from_reasoning(reasoning: ReasoningConfig | None) -> dict[str, Any]:
    if reasoning is None:
        return {}
    return {
        "thinking_level": reasoning.thinking_level,
        "thinking_budget": reasoning.thinking_budget,
    }


def _reasoning_effort_from_reasoning(reasoning: ReasoningConfig | None) -> str | None:
    if reasoning is None:
        return None
    return reasoning.reasoning_effort


def _anthropic_max_tokens_from_reasoning(reasoning: ReasoningConfig | None) -> int:
    thinking_budget = reasoning.thinking_budget if reasoning is not None else None
    if thinking_budget is None:
        return ANTHROPIC_DEFAULT_MAX_TOKENS
    return max(ANTHROPIC_DEFAULT_MAX_TOKENS, thinking_budget + ANTHROPIC_THINKING_OUTPUT_BUFFER)


def _anthropic_thinking_payload(reasoning: ReasoningConfig | None) -> dict[str, Any] | None:
    if reasoning is None or reasoning.thinking_budget is None:
        return None
    return {
        "type": "enabled",
        "budget_tokens": reasoning.thinking_budget,
    }


def _anthropic_text_block(text: str, cache: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "text",
        "text": text,
    }
    if cache:
        block["cache_control"] = dict(ANTHROPIC_EPHEMERAL_CACHE_CONTROL)
    return block


def _anthropic_message(role: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": role,
        "content": _copy_content_blocks(blocks),
    }


def _anthropic_system_blocks(system_prompt: str, cache: bool = False) -> list[dict[str, Any]]:
    if not system_prompt:
        return []
    return [_anthropic_text_block(system_prompt, cache=cache)]


class GeminiClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def create_chat_completion(self, model: str, system_prompt: str, user_prompt: str) -> str:
        return self.generate_content_messages(
            model=model,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

    def generate_content(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        reasoning: ReasoningConfig | None = None,
    ) -> str:
        return self.generate_content_messages(
            model=model,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            reasoning=reasoning,
        )

    def generate_content_messages(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        reasoning: ReasoningConfig | None = None,
    ) -> str:
        contents = []
        for message in _normalize_chat_messages(messages):
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message["content"]}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": _build_generate_content_config(**_thinking_kwargs_from_reasoning(reasoning)),
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        def _do_generate() -> str:
            response = _post_json(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                payload,
                {"x-goog-api-key": self.api_key},
            )
            try:
                parts = response["candidates"][0]["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts).strip()
                if not text:
                    raise KeyError("empty text")
                return text
            except (KeyError, IndexError, TypeError) as exc:
                raise RetriableContentError(f"Unexpected Gemini response: {json.dumps(response)}") from exc

        return _with_content_retry(_do_generate)

    def create_interaction(
        self,
        model: str,
        user_input: str,
        system_instruction: str = "",
        previous_interaction_id: str | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": user_input,
            "generation_config": _build_interaction_generation_config(
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
            ),
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction
        if previous_interaction_id is not None:
            payload["previous_interaction_id"] = previous_interaction_id

        def _do_interact() -> dict[str, Any]:
            response = _post_json(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                payload,
                {"x-goog-api-key": self.api_key},
            )
            if not isinstance(response, dict) or not response.get("id"):
                raise RetriableContentError(f"Unexpected Gemini interaction response: {json.dumps(response)}")
            _extract_interaction_text(response)  # validates content; raises RetriableContentError
            return response

        return _with_content_retry(_do_interact)

    def extract_interaction_text(self, interaction: dict[str, Any]) -> str:
        return _extract_interaction_text(interaction)


class GeminiInteractionSession:
    def __init__(self, client: GeminiClient, model: str, system_prompt: str, initial_user_prompt: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.initial_user_prompt = initial_user_prompt
        self._turn_count = 0
        self._previous_interaction_id: str | None = None
        self.last_call_metadata: dict[str, Any] = {}

    @property
    def session_mode(self) -> str:
        return "gemini-interactions-previous-interaction-id"

    def generate_turn(
        self,
        turn_prompt: str,
        reasoning: ReasoningConfig | None = None,
    ) -> tuple[str, str, str, str | None]:
        effective_user_prompt = turn_prompt
        if self._turn_count == 0:
            effective_user_prompt = f"{self.initial_user_prompt}\n\n{turn_prompt}"

        previous_interaction_id = self._previous_interaction_id
        interaction = self.client.create_interaction(
            model=self.model,
            user_input=effective_user_prompt,
            system_instruction=self.system_prompt,
            previous_interaction_id=previous_interaction_id,
            **_thinking_kwargs_from_reasoning(reasoning),
        )
        raw_text = self.client.extract_interaction_text(interaction)
        interaction_id = str(interaction["id"])
        self._previous_interaction_id = interaction_id
        self._turn_count += 1
        self.last_call_metadata = {
            "request_id": interaction_id,
            "previous_request_id": previous_interaction_id,
            "interaction_id": interaction_id,
            "previous_interaction_id": previous_interaction_id,
        }
        return raw_text, effective_user_prompt, interaction_id, previous_interaction_id


class OpenAIClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def generate_content(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        reasoning: ReasoningConfig | None = None,
    ) -> str:
        response = self.create_response(
            model=model,
            input_text=user_prompt,
            instructions=system_prompt,
            reasoning_effort=_reasoning_effort_from_reasoning(reasoning),
        )
        return self.extract_response_text(response)

    def create_response(
        self,
        model: str,
        input_text: str,
        instructions: str = "",
        previous_response_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_text}],
                }
            ],
        }
        if instructions:
            payload["instructions"] = instructions
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}

        def _do_respond() -> dict[str, Any]:
            response = _post_json(
                "https://api.openai.com/v1/responses",
                payload,
                {"Authorization": f"Bearer {self.api_key}"},
            )
            if not isinstance(response, dict) or not response.get("id"):
                raise RetriableContentError(f"Unexpected OpenAI response: {json.dumps(response)}")
            _extract_openai_response_text(response)  # validates content; raises RetriableContentError
            return response

        return _with_content_retry(_do_respond)

    def extract_response_text(self, response: dict[str, Any]) -> str:
        return _extract_openai_response_text(response)


class AnthropicClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def generate_content(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        reasoning: ReasoningConfig | None = None,
    ) -> str:
        response = self.create_message(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [_anthropic_text_block(user_prompt)],
                }
            ],
            system_prompt=system_prompt,
            reasoning=reasoning,
            cache_system_prompt=True,
        )
        return self.extract_message_text(response)

    def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        reasoning: ReasoningConfig | None = None,
        cache_system_prompt: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": _anthropic_max_tokens_from_reasoning(reasoning),
            "messages": messages,
        }
        system_blocks = _anthropic_system_blocks(system_prompt, cache=cache_system_prompt)
        if system_blocks:
            payload["system"] = system_blocks
        thinking = _anthropic_thinking_payload(reasoning)
        if thinking is not None:
            payload["thinking"] = thinking

        def _do_message() -> dict[str, Any]:
            response = _post_json(
                "https://api.anthropic.com/v1/messages",
                payload,
                {
                    "x-api-key": self.api_key,
                    "anthropic-version": ANTHROPIC_API_VERSION,
                },
            )
            if not isinstance(response, dict) or not response.get("id"):
                raise RetriableContentError(f"Unexpected Anthropic response: {json.dumps(response)}")
            _extract_anthropic_message_text(response)
            return response

        return _with_content_retry(_do_message)

    def extract_message_text(self, response: dict[str, Any]) -> str:
        return _extract_anthropic_message_text(response)


class OpenAIResponsesSession:
    def __init__(self, client: OpenAIClient, model: str, system_prompt: str, initial_user_prompt: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.initial_user_prompt = initial_user_prompt
        self._turn_count = 0
        self._previous_response_id: str | None = None
        self.last_call_metadata: dict[str, Any] = {}

    @property
    def session_mode(self) -> str:
        return "openai-responses-previous-response-id"

    def generate_turn(
        self,
        turn_prompt: str,
        reasoning: ReasoningConfig | None = None,
    ) -> tuple[str, str, str, str | None]:
        effective_user_prompt = turn_prompt
        if self._turn_count == 0:
            effective_user_prompt = f"{self.initial_user_prompt}\n\n{turn_prompt}"

        previous_response_id = self._previous_response_id
        response = self.client.create_response(
            model=self.model,
            input_text=effective_user_prompt,
            instructions=self.system_prompt,
            previous_response_id=previous_response_id,
            reasoning_effort=_reasoning_effort_from_reasoning(reasoning),
        )
        raw_text = self.client.extract_response_text(response)
        response_id = str(response["id"])
        self._previous_response_id = response_id
        self._turn_count += 1
        self.last_call_metadata = {
            "request_id": response_id,
            "previous_request_id": previous_response_id,
            "response_id": response_id,
            "previous_response_id": previous_response_id,
        }
        return raw_text, effective_user_prompt, response_id, previous_response_id


class AnthropicMessagesSession:
    def __init__(self, client: AnthropicClient, model: str, system_prompt: str, initial_user_prompt: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.initial_user_prompt = initial_user_prompt
        self._turn_count = 0
        self._history_messages: list[dict[str, Any]] = []
        self.last_call_metadata: dict[str, Any] = {}

    @property
    def session_mode(self) -> str:
        return "anthropic-messages-prompt-caching"

    def _history_with_cache_breakpoint(self) -> list[dict[str, Any]]:
        history = [
            {
                "role": str(message["role"]),
                "content": _copy_content_blocks(list(message["content"])),
            }
            for message in self._history_messages
        ]
        if not history:
            return history
        last_message = history[-1]
        content = last_message.get("content")
        if not isinstance(content, list) or not content:
            return history
        content[-1]["cache_control"] = dict(ANTHROPIC_EPHEMERAL_CACHE_CONTROL)
        return history

    def generate_turn(
        self,
        turn_prompt: str,
        reasoning: ReasoningConfig | None = None,
    ) -> tuple[str, str, str, str | None]:
        user_blocks = [_anthropic_text_block(turn_prompt)]
        effective_user_prompt = turn_prompt
        if self._turn_count == 0:
            effective_user_prompt = f"{self.initial_user_prompt}\n\n{turn_prompt}"
            user_blocks = [
                _anthropic_text_block(self.initial_user_prompt, cache=True),
                _anthropic_text_block(f"\n\n{turn_prompt}"),
            ]

        payload_messages = self._history_with_cache_breakpoint()
        payload_messages.append(_anthropic_message("user", user_blocks))
        response = self.client.create_message(
            model=self.model,
            messages=payload_messages,
            system_prompt=self.system_prompt,
            reasoning=reasoning,
        )
        raw_text = self.client.extract_message_text(response)
        message_id = str(response["id"])
        self._history_messages.append(_anthropic_message("user", user_blocks))
        self._history_messages.append(_anthropic_message("assistant", [_anthropic_text_block(raw_text)]))
        self._turn_count += 1

        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        self.last_call_metadata = {
            "request_id": message_id,
            "previous_request_id": None,
            "message_id": message_id,
            "history_message_count": len(self._history_messages),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }
        return raw_text, effective_user_prompt, message_id, None
