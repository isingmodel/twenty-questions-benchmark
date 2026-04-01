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


RETRIABLE_HTTP_CODES = {408, 429, 499, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BASE_DELAY_SECONDS = 1.5
DEFAULT_MAX_JITTER_SECONDS = 0.65
DEFAULT_PRE_REQUEST_DELAY_SECONDS = 1.5
RETRY_DELAY_SECONDS = 4.


def _normalize_chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"])
        normalized.append({"role": role, "content": content})
    return normalized


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


def _extract_interaction_text(interaction: dict[str, Any]) -> str:
    outputs = interaction.get("outputs")
    if not isinstance(outputs, list):
        raise HTTPError(f"Unexpected Gemini interaction response: {json.dumps(interaction)}")
    for output in reversed(outputs):
        if not isinstance(output, dict):
            continue
        text = output.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    raise HTTPError(f"Gemini interaction did not return text output: {json.dumps(interaction)}")


def _extract_openai_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = response.get("output")
    if not isinstance(outputs, list):
        raise HTTPError(f"Unexpected OpenAI response: {json.dumps(response)}")

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
    raise HTTPError(f"OpenAI response did not return text output: {json.dumps(response)}")


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
            raise HTTPError(f"Unexpected Gemini response: {json.dumps(response)}") from exc

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

        response = _post_json(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            payload,
            {"x-goog-api-key": self.api_key},
        )
        if not isinstance(response, dict) or not response.get("id"):
            raise HTTPError(f"Unexpected Gemini interaction response: {json.dumps(response)}")
        _extract_interaction_text(response)
        return response

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

        response = _post_json(
            "https://api.openai.com/v1/responses",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        if not isinstance(response, dict) or not response.get("id"):
            raise HTTPError(f"Unexpected OpenAI response: {json.dumps(response)}")
        _extract_openai_response_text(response)
        return response

    def extract_response_text(self, response: dict[str, Any]) -> str:
        return _extract_openai_response_text(response)


class OpenAIResponsesSession:
    def __init__(self, client: OpenAIClient, model: str, system_prompt: str, initial_user_prompt: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.initial_user_prompt = initial_user_prompt
        self._turn_count = 0
        self._previous_response_id: str | None = None

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
        return raw_text, effective_user_prompt, response_id, previous_response_id
