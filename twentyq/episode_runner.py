from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .clients import (
    AnthropicClient,
    AnthropicMessagesSession,
    GeminiClient,
    GeminiInteractionSession,
    HTTPError,
    OpenAIClient,
    OpenAIResponsesSession,
)
from .env import get_required_env, get_required_env_any
from .prompts import (
    DEFAULT_GUESSER_PROMPT_SET,
    ROOT,
    load_guesser_prompts,
    load_prompt,
    render_template,
)
from .reasoning import ReasoningConfig
from .run_logs import RunLogger


PROVIDER = "gemini"
OPENAI_PROVIDER = "openai"
ANTHROPIC_PROVIDER = "anthropic"
THINKING_LEVEL_CHOICES = ("minimal", "low", "medium", "high")
REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high")
DEFAULT_TARGET_ID = "object_toothbrush"
DEFAULT_BUDGET = 20
DEFAULT_GUESSER_MODEL = "gemini-2.5-flash"
DEFAULT_JUDGE_MODEL = "gemini-3-flash-preview"
JUDGMENT_TRUE = "Yes"
JUDGMENT_FALSE = "No"
JUDGMENT_AMBIGUOUS = "Ambiguous"
QUESTION_TYPE_IDENTITY = "identity"
QUESTION_TYPE_NON_IDENTITY = "non_identity"
GEMINI_25_THINKING_BUDGET_BY_EFFORT = {
    "minimal": 1024,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}
GEMINI_25_FLASH_THINKING_LEVEL_BY_EFFORT = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
GEMINI_25_PRO_THINKING_LEVEL_BY_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
}
GEMINI_3_FLASH_THINKING_LEVEL_BY_EFFORT = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
GEMINI_3_PRO_THINKING_LEVEL_BY_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
}
GEMINI_31_PRO_THINKING_LEVEL_BY_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
CLAUDE_THINKING_BUDGET_BY_EFFORT = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 24576,
}
OPENAI_REASONING_EFFORT_IDENTITY = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


@dataclass(frozen=True)
class ModelReasoningCapability:
    provider: str
    prefixes: tuple[str, ...]
    supports_thinking_level: bool
    supports_thinking_budget: bool
    supported_levels: tuple[str, ...] | None = None
    min_thinking_budget: int | None = None
    effort_to_level: dict[str, str] | None = None
    effort_to_budget: dict[str, int] | None = None


MODEL_PROVIDER_PREFIXES = (
    (PROVIDER, ("gemini-",)),
    (OPENAI_PROVIDER, ("gpt-", "o1", "o3", "o4", "codex-")),
    (ANTHROPIC_PROVIDER, ("claude-",)),
)

MODEL_REASONING_CAPABILITIES = (
    ModelReasoningCapability(
        provider=OPENAI_PROVIDER,
        prefixes=("gpt-", "o1", "o3", "o4", "codex-"),
        supports_thinking_level=True,
        supports_thinking_budget=False,
        supported_levels=THINKING_LEVEL_CHOICES,
        effort_to_level=OPENAI_REASONING_EFFORT_IDENTITY,
    ),
    ModelReasoningCapability(
        provider=ANTHROPIC_PROVIDER,
        prefixes=("claude-opus-4", "claude-sonnet-4", "claude-3-7-sonnet"),
        supports_thinking_level=False,
        supports_thinking_budget=True,
        min_thinking_budget=1024,
        effort_to_budget=CLAUDE_THINKING_BUDGET_BY_EFFORT,
    ),
    ModelReasoningCapability(
        provider=PROVIDER,
        prefixes=("gemini-2.5-flash-lite", "gemini-2.5-flash"),
        supports_thinking_level=True,
        supports_thinking_budget=True,
        supported_levels=THINKING_LEVEL_CHOICES,
        effort_to_level=GEMINI_25_FLASH_THINKING_LEVEL_BY_EFFORT,
    ),
    ModelReasoningCapability(
        provider=PROVIDER,
        prefixes=("gemini-2.5-pro",),
        supports_thinking_level=True,
        supports_thinking_budget=True,
        effort_to_level=GEMINI_25_PRO_THINKING_LEVEL_BY_EFFORT,
    ),
    ModelReasoningCapability(
        provider=PROVIDER,
        prefixes=("gemini-3.1-flash-lite",),
        supports_thinking_level=True,
        supports_thinking_budget=False,
        supported_levels=THINKING_LEVEL_CHOICES,
        effort_to_level=GEMINI_3_FLASH_THINKING_LEVEL_BY_EFFORT,
    ),
    ModelReasoningCapability(
        provider=PROVIDER,
        prefixes=("gemini-3-flash",),
        supports_thinking_level=True,
        supports_thinking_budget=False,
        supported_levels=THINKING_LEVEL_CHOICES,
        effort_to_level=GEMINI_3_FLASH_THINKING_LEVEL_BY_EFFORT,
    ),
    ModelReasoningCapability(
        provider=PROVIDER,
        prefixes=("gemini-3.1-pro",),
        supports_thinking_level=True,
        supports_thinking_budget=False,
        supported_levels=("low", "medium", "high"),
        effort_to_level=GEMINI_31_PRO_THINKING_LEVEL_BY_EFFORT,
    ),
    ModelReasoningCapability(
        provider=PROVIDER,
        prefixes=("gemini-3-pro",),
        supports_thinking_level=True,
        supports_thinking_budget=False,
        supported_levels=("low", "high"),
        effort_to_level=GEMINI_3_PRO_THINKING_LEVEL_BY_EFFORT,
    ),
)


@dataclass(frozen=True)
class JudgeDecision:
    label: str
    reason: str | None = None
    question_type: str | None = None
    direct_target_guess: bool | None = None


@dataclass
class FullGameConfig:
    target_id: str
    budget: int
    guesser_model: str
    judge_model: str
    guesser_reasoning: ReasoningConfig
    judge_reasoning: ReasoningConfig
    run_dir: Path | None
    guesser_prompt_set: str = DEFAULT_GUESSER_PROMPT_SET
    guesser_initial_prompt_path: Path | None = None
    guesser_turn_prompt_path: Path | None = None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def provider_for_model(model: str) -> str:
    normalized = _normalize_model_name(model)
    for provider, prefixes in MODEL_PROVIDER_PREFIXES:
        if _matches_any_prefix(normalized, prefixes):
            return provider
    raise ValueError(f"Unsupported model provider for model {model!r}")


def _normalize_model_name(model: str) -> str:
    return model.strip().lower()


def _matches_any_prefix(normalized_model: str, prefixes: tuple[str, ...]) -> bool:
    return normalized_model.startswith(prefixes)


def _find_reasoning_capability(model: str) -> ModelReasoningCapability | None:
    normalized = _normalize_model_name(model)
    for capability in MODEL_REASONING_CAPABILITIES:
        if _matches_any_prefix(normalized, capability.prefixes):
            return capability
    return None


def _format_supported_levels(levels: tuple[str, ...]) -> str:
    if len(levels) == 1:
        return f"'{levels[0]}'"
    if len(levels) == 2:
        return f"'{levels[0]}' and '{levels[1]}'"
    head = ", ".join(f"'{level}'" for level in levels[:-1])
    return f"{head}, and '{levels[-1]}'"


def _create_gemini_client() -> GeminiClient:
    return GeminiClient(get_required_env("gemini_key"))


def _create_openai_client() -> OpenAIClient:
    api_key = get_required_env_any("OPENAI_API_KEY", "openai_key")
    return OpenAIClient(api_key)


def _create_anthropic_client() -> AnthropicClient:
    api_key = get_required_env_any("CLAUDE_API_KEY", "ANTHROPIC_API_KEY", "anthropic_key")
    return AnthropicClient(api_key)


def _create_client_for_model(model: str) -> Any:
    provider = provider_for_model(model)
    if provider == PROVIDER:
        return _create_gemini_client()
    if provider == OPENAI_PROVIDER:
        return _create_openai_client()
    return _create_anthropic_client()


def _create_guesser_session(client: Any, model: str, system_prompt: str, initial_user_prompt: str) -> Any:
    provider = provider_for_model(model)
    if provider == PROVIDER:
        return GeminiInteractionSession(
            client=client,
            model=model,
            system_prompt=system_prompt,
            initial_user_prompt=initial_user_prompt,
        )
    if provider == OPENAI_PROVIDER:
        return OpenAIResponsesSession(
            client=client,
            model=model,
            system_prompt=system_prompt,
            initial_user_prompt=initial_user_prompt,
        )
    return AnthropicMessagesSession(
        client=client,
        model=model,
        system_prompt=system_prompt,
        initial_user_prompt=initial_user_prompt,
    )


def _call_stateless_model(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning: ReasoningConfig,
) -> tuple[str, int]:
    return _call_model(
        client,
        "generate_content",
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=reasoning,
    )


def _call_model(client: Any, method_name: str, **kwargs: Any) -> tuple[Any, int]:
    started = perf_counter()
    value = getattr(client, method_name)(**kwargs)
    latency_ms = int((perf_counter() - started) * 1000)
    return value, latency_ms


def _normalize_judge_output(raw_output: str) -> str:
    normalized = re.sub(r"[`*_]+", "", str(raw_output)).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        raise ValueError("Judge returned empty output")

    first_token = re.sub(r"[^a-z]+", "", normalized.split()[0])
    if first_token in {"true", "yes", "o"}:
        return JUDGMENT_TRUE
    if first_token in {"false", "no", "x"}:
        return JUDGMENT_FALSE
    if first_token in {"ambiguous", "invalid", "unknown", "unclear"}:
        return JUDGMENT_AMBIGUOUS
    raise ValueError(f"Unrecognized judge output: {raw_output!r}")


def _normalize_judge_question_type(raw_value: Any) -> str:
    normalized = str(raw_value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"identity", "identity_check", "direct_guess", "direct_identity"}:
        return QUESTION_TYPE_IDENTITY
    if normalized in {"non_identity", "attribute", "descriptive", "category"}:
        return QUESTION_TYPE_NON_IDENTITY
    raise ValueError(f"Unrecognized judge question_type: {raw_value!r}")


def _parse_bool_field(raw_value: Any, field_name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    normalized = str(raw_value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"Unrecognized boolean value for {field_name}: {raw_value!r}")


def _parse_judge_response(raw_output: str, *, require_structured: bool = False) -> JudgeDecision:
    raw_text = str(raw_output).strip()
    if not raw_text:
        raise ValueError("Judge returned empty output")

    label_source = raw_text
    reason: str | None = None
    question_type: str | None = None
    direct_target_guess: bool | None = None
    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            label_source = str(parsed.get("label") or parsed.get("judgment") or parsed.get("verdict") or "")
            raw_reason = parsed.get("reason")
            if raw_reason is not None:
                normalized_reason = str(raw_reason).strip()
                reason = normalized_reason or None
            raw_question_type = parsed.get("question_type", parsed.get("question_kind"))
            if raw_question_type is not None:
                question_type = _normalize_judge_question_type(raw_question_type)
            raw_direct_target_guess = parsed.get(
                "direct_target_guess",
                parsed.get("identity_match", parsed.get("is_correct_guess")),
            )
            if raw_direct_target_guess is not None:
                direct_target_guess = _parse_bool_field(raw_direct_target_guess, "direct_target_guess")
    elif require_structured:
        raise ValueError(f"Judge must return a JSON object, got: {raw_output!r}")

    label = _normalize_judge_output(label_source)
    if require_structured and question_type is None:
        raise ValueError(f"Judge response missing question_type: {raw_output!r}")
    if require_structured and direct_target_guess is None:
        raise ValueError(f"Judge response missing direct_target_guess: {raw_output!r}")
    if direct_target_guess is True and question_type != QUESTION_TYPE_IDENTITY:
        raise ValueError("Judge marked a non-identity question as a direct target guess")
    if direct_target_guess is True and label != JUDGMENT_TRUE:
        raise ValueError("Judge marked a direct target guess as solved without label Yes")

    return JudgeDecision(
        label=label,
        reason=reason,
        question_type=question_type,
        direct_target_guess=direct_target_guess,
    )


def _classify_error_message(error_message: str) -> str:
    normalized = error_message.lower()
    transient_markers = [
        "http 408",
        "http 429",
        "http 499",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "network error",
        "socket timeout",
        "unavailable",
        "high demand",
    ]
    for marker in transient_markers:
        if marker in normalized:
            return "transient_error"
    return "runtime_error"


def _write_episode_markdown(logger: RunLogger, episode: dict[str, Any]) -> None:
    lines = [
        f"# Full Game Episode {episode['target_id']}",
        "",
        f"- target_name: {episode['target_name']}",
        f"- solved: {episode['outcome']['solved']}",
        f"- turns_used: {episode['outcome']['turns_used']}",
    ]
    for turn in episode["turns"]:
        lines.extend(
            [
                "",
                f"Q{turn['turn']}: {turn['question']}",
                f"A{turn['turn']}: {turn['judgment']}",
            ]
        )
    (logger.episodes_dir / f"{episode['target_id']}.md").write_text("\n".join(lines), encoding="utf-8")


def _relative_run_dir(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _build_judge_user_prompt(
    judge_turn_template: str,
    target: dict[str, Any],
    question: str,
) -> str:
    return render_template(
        judge_turn_template,
        target=json.dumps(target, ensure_ascii=True, sort_keys=True),
        question=question,
    )


def _validate_budget(budget: int) -> int:
    if budget < 1:
        raise ValueError(f"Budget must be positive, got {budget}")
    return budget


def _validate_reasoning_config(
    role: str,
    model: str,
    thinking_level: str | None,
    thinking_budget: int | None,
) -> ReasoningConfig:
    if thinking_level and thinking_budget is not None:
        raise ValueError(f"{role} cannot set both thinking level and thinking budget")
    if thinking_budget is not None and thinking_budget < 0:
        raise ValueError(f"{role} thinking budget must be non-negative, got {thinking_budget}")

    provider = provider_for_model(model)
    capability = _find_reasoning_capability(model)
    if provider == OPENAI_PROVIDER:
        if thinking_budget is not None:
            raise ValueError(f"{role} model {model!r} does not support thinking budgets")
        if thinking_level is None:
            return ReasoningConfig()
        supported_levels = capability.supported_levels if capability is not None else THINKING_LEVEL_CHOICES
        if supported_levels is not None and thinking_level not in supported_levels:
            raise ValueError(f"{role} model {model!r} has unsupported thinking level {thinking_level!r}")
        return ReasoningConfig(reasoning_effort=thinking_level)
    if provider == ANTHROPIC_PROVIDER:
        if thinking_level is not None:
            raise ValueError(f"{role} model {model!r} uses thinking budgets, not thinking levels")
        if thinking_budget is None:
            return ReasoningConfig()
        if capability is None or not capability.supports_thinking_budget:
            raise ValueError(f"{role} model {model!r} does not support extended thinking budgets")
        min_budget = capability.min_thinking_budget or 0
        if thinking_budget < min_budget:
            raise ValueError(f"{role} model {model!r} requires thinking budgets of at least 1024 tokens")
        return ReasoningConfig(thinking_budget=thinking_budget)

    if capability is not None:
        if thinking_budget is not None and not capability.supports_thinking_budget:
            raise ValueError(f"{role} model {model!r} uses thinking levels, not thinking budgets")
        if thinking_level is not None and not capability.supports_thinking_level:
            raise ValueError(f"{role} model {model!r} does not support thinking levels")
        if capability.supported_levels is not None and thinking_level not in {None, *capability.supported_levels}:
            supported_text = _format_supported_levels(capability.supported_levels)
            raise ValueError(f"{role} model {model!r} only supports thinking levels {supported_text}")
    return ReasoningConfig(thinking_level=thinking_level, thinking_budget=thinking_budget)


def _reasoning_to_payload(reasoning: ReasoningConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if reasoning.thinking_level is not None:
        payload["thinking_level"] = reasoning.thinking_level
    if reasoning.thinking_budget is not None:
        payload["thinking_budget"] = reasoning.thinking_budget
    if reasoning.reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning.reasoning_effort
    return payload


def resolve_reasoning_effort(model: str, effort: str | None, role: str = "Model") -> ReasoningConfig:
    if effort is None:
        return ReasoningConfig()

    normalized_effort = effort.strip().lower()
    if normalized_effort not in REASONING_EFFORT_CHOICES:
        raise ValueError(f"Unsupported reasoning effort {effort!r}")

    capability = _find_reasoning_capability(model)
    if capability is None:
        raise ValueError(f"Unsupported model family for reasoning effort mapping: {model!r}")
    if capability.effort_to_level is not None:
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=capability.effort_to_level[normalized_effort],
            thinking_budget=None,
        )
    if capability.effort_to_budget is not None:
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=None,
            thinking_budget=capability.effort_to_budget[normalized_effort],
        )
    raise ValueError(f"Unsupported model family for reasoning effort mapping: {model!r}")


def run_full_game_episode(
    config: FullGameConfig,
    target: dict[str, Any],
    runs_dir: Path,
    progress_callback: Any = None,
) -> tuple[int, dict[str, Any]]:
    guesser_provider = provider_for_model(config.guesser_model)
    judge_provider = provider_for_model(config.judge_model)
    guesser_client = _create_client_for_model(config.guesser_model)
    judge_client = _create_client_for_model(config.judge_model)
    logger = RunLogger.create(
        runs_dir=runs_dir,
        split="full-game-test",
        guesser_provider=guesser_provider,
        guesser_model=config.guesser_model,
    )
    guesser_prompts = load_guesser_prompts(
        prompt_set=config.guesser_prompt_set,
        initial_prompt_path=config.guesser_initial_prompt_path,
        turn_prompt_path=config.guesser_turn_prompt_path,
    )
    guesser_initial = guesser_prompts.initial_prompt
    guesser_turn_template = guesser_prompts.turn_prompt
    judge_system = load_prompt("judge-system.txt")
    judge_turn_template = load_prompt("judge-turn-template.txt")

    logger.write_json(
        "run_config.json",
        {
            "run_id": logger.run_id,
            "mode": "full-game-test",
            "config": {
                "target_id": config.target_id,
                "budget": config.budget,
                "guesser_provider": guesser_provider,
                "guesser_model": config.guesser_model,
                "guesser_reasoning": _reasoning_to_payload(config.guesser_reasoning),
                "guesser_prompt_set": guesser_prompts.name,
                "guesser_prompt_source": guesser_prompts.source,
                "guesser_initial_prompt_path": guesser_prompts.initial_prompt_path,
                "guesser_turn_prompt_path": guesser_prompts.turn_prompt_path,
                "judge_provider": judge_provider,
                "judge_model": config.judge_model,
                "judge_reasoning": _reasoning_to_payload(config.judge_reasoning),
                "run_dir": _relative_run_dir(logger.root_dir),
            },
        },
    )
    logger.log_event(
        {
            "ts": _utc_now(),
            "event": "run_started",
            "run_id": logger.run_id,
            "mode": "full-game-test",
            "config": {
                "target_id": config.target_id,
                "budget": config.budget,
                "guesser_provider": guesser_provider,
                "guesser_model": config.guesser_model,
                "guesser_reasoning": _reasoning_to_payload(config.guesser_reasoning),
                "guesser_prompt_set": guesser_prompts.name,
                "guesser_prompt_source": guesser_prompts.source,
                "guesser_initial_prompt_path": guesser_prompts.initial_prompt_path,
                "guesser_turn_prompt_path": guesser_prompts.turn_prompt_path,
                "judge_provider": judge_provider,
                "judge_model": config.judge_model,
                "judge_reasoning": _reasoning_to_payload(config.judge_reasoning),
            },
        }
    )
    if progress_callback is not None:
        progress_callback({"event": "run_started", "run_id": logger.run_id, "target_id": target["id"], "turn": 0})

    turns: list[dict[str, Any]] = []
    outcome = {
        "solved": False,
        "turns_used": 0,
        "final_question": None,
        "final_question_correct": False,
    }
    guesser_session = _create_guesser_session(
        client=guesser_client,
        model=config.guesser_model,
        system_prompt="",
        initial_user_prompt=guesser_initial,
    )
    judge_session_mode = "stateless-generate-content"

    try:
        for turn_number in range(1, config.budget + 1):
            previous_feedback = "(none yet)"
            if turns:
                previous_feedback = (
                    f"Your previous question was: {turns[-1]['question']}\n"
                    f"Judge reply: {turns[-1]['judgment']}"
                )
            turn_prompt = render_template(
                guesser_turn_template,
                previous_feedback=previous_feedback,
            )
            guesser_result, guesser_latency_ms = _call_model(
                guesser_session,
                "generate_turn",
                turn_prompt=turn_prompt,
                reasoning=config.guesser_reasoning,
            )
            (
                guesser_raw_output,
                guesser_user_prompt,
                guesser_request_id,
                guesser_previous_request_id,
            ) = guesser_result
            guesser_call_metadata = {}
            if hasattr(guesser_session, "last_call_metadata"):
                guesser_call_metadata = dict(getattr(guesser_session, "last_call_metadata"))
            logger.log_event(
                {
                    "ts": _utc_now(),
                    "event": "model_call",
                    "role": "guesser",
                    "turn": turn_number,
                    "target_id": target["id"],
                    "provider": guesser_provider,
                    "model": config.guesser_model,
                    "latency_ms": guesser_latency_ms,
                    "session_mode": guesser_session.session_mode,
                    "system_prompt": "",
                    "prompt_set": guesser_prompts.name,
                    "prompt_source": guesser_prompts.source,
                    "initial_prompt_path": guesser_prompts.initial_prompt_path,
                    "turn_prompt_path": guesser_prompts.turn_prompt_path,
                    "user_prompt": guesser_user_prompt,
                    "turn_prompt": turn_prompt,
                    "raw_output": guesser_raw_output,
                    "generation_config": _reasoning_to_payload(config.guesser_reasoning),
                    "request_id": guesser_request_id,
                    "previous_request_id": guesser_previous_request_id,
                    **guesser_call_metadata,
                }
            )

            judge_user_prompt = _build_judge_user_prompt(
                judge_turn_template=judge_turn_template,
                target=target,
                question=guesser_raw_output,
            )
            judge_raw_output, judge_latency_ms = _call_stateless_model(
                judge_client,
                model=config.judge_model,
                system_prompt=judge_system,
                user_prompt=judge_user_prompt,
                reasoning=config.judge_reasoning,
            )
            judge_decision = _parse_judge_response(judge_raw_output, require_structured=True)
            judgment = judge_decision.label
            logger.log_event(
                {
                    "ts": _utc_now(),
                    "event": "model_call",
                    "role": "judge",
                    "turn": turn_number,
                    "target_id": target["id"],
                    "provider": judge_provider,
                    "model": config.judge_model,
                    "latency_ms": judge_latency_ms,
                    "session_mode": judge_session_mode,
                    "system_prompt": judge_system,
                    "user_prompt": judge_user_prompt,
                    "raw_output": judge_raw_output,
                    "normalized_output": judgment,
                    "judge_reason": judge_decision.reason,
                    "judge_question_type": judge_decision.question_type,
                    "judge_direct_target_guess": judge_decision.direct_target_guess,
                    "generation_config": _reasoning_to_payload(config.judge_reasoning),
                }
            )

            turns.append(
                {
                    "turn": turn_number,
                    "question": guesser_raw_output,
                    "judgment": judgment,
                    "judge_reason": judge_decision.reason,
                    "guesser_provider": guesser_provider,
                    "guesser_model": config.guesser_model,
                    "judge_provider": judge_provider,
                    "judge_model": config.judge_model,
                    "guesser_latency_ms": guesser_latency_ms,
                    "judge_latency_ms": judge_latency_ms,
                    "guesser_raw_output": guesser_raw_output,
                    "judge_raw_output": judge_raw_output,
                    "judge_question_type": judge_decision.question_type,
                    "judge_direct_target_guess": judge_decision.direct_target_guess,
                    "guesser_session_mode": guesser_session.session_mode,
                    "judge_session_mode": judge_session_mode,
                    "guesser_request_id": guesser_request_id,
                    "guesser_previous_request_id": guesser_previous_request_id,
                    **guesser_call_metadata,
                }
            )
            outcome["turns_used"] = turn_number
            outcome["final_question"] = guesser_raw_output
            outcome["final_question_correct"] = bool(judge_decision.direct_target_guess)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "turn_completed",
                        "run_id": logger.run_id,
                        "target_id": target["id"],
                        "turn": turn_number,
                        "question": guesser_raw_output,
                        "judgment": judgment,
                        "solved": outcome["final_question_correct"],
                    }
                )
            if outcome["final_question_correct"]:
                outcome["solved"] = True
                break

        episode = {
            "mode": "full-game-test",
            "target_id": target["id"],
            "target_name": target["name"],
            "target_domain": target["domain"],
            "budget": config.budget,
            "guesser_prompt_set": guesser_prompts.name,
            "guesser_prompt_source": guesser_prompts.source,
            "guesser_initial_prompt_path": guesser_prompts.initial_prompt_path,
            "guesser_turn_prompt_path": guesser_prompts.turn_prompt_path,
            "turns": turns,
            "outcome": outcome,
        }
        logger.write_json(f"episodes/{target['id']}.json", episode)
        _write_episode_markdown(logger, episode)
        summary = {
            "run_id": logger.run_id,
            "mode": "full-game-test",
            "target_id": target["id"],
            "target_name": target["name"],
            "guesser_prompt_set": guesser_prompts.name,
            "guesser_prompt_source": guesser_prompts.source,
            "guesser_initial_prompt_path": guesser_prompts.initial_prompt_path,
            "guesser_turn_prompt_path": guesser_prompts.turn_prompt_path,
            "solved": outcome["solved"],
            "turns_used": outcome["turns_used"],
            "final_question": outcome["final_question"],
            "final_question_correct": outcome["final_question_correct"],
            "run_dir": str(logger.root_dir),
        }
        logger.write_json("summary.json", summary)
        logger.log_event({"ts": _utc_now(), "event": "run_completed", "run_id": logger.run_id, "summary": summary})
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "run_completed",
                    "run_id": logger.run_id,
                    "target_id": target["id"],
                    "turn": outcome["turns_used"],
                    "solved": outcome["solved"],
                }
            )
        return 0, summary
    except (HTTPError, RuntimeError, ValueError) as exc:
        error_type = _classify_error_message(str(exc))
        logger.log_event(
            {
                "ts": _utc_now(),
                "event": "run_failed",
                "run_id": logger.run_id,
                "target_id": target["id"],
                "turn": len(turns),
                "error": str(exc),
                "error_type": error_type,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "run_failed",
                    "run_id": logger.run_id,
                    "target_id": target["id"],
                    "turn": len(turns),
                    "error": str(exc),
                }
            )
        return 1, {
            "run_id": logger.run_id,
            "mode": "full-game-test",
            "target_id": target["id"],
            "target_name": target["name"],
            "guesser_prompt_set": guesser_prompts.name,
            "guesser_prompt_source": guesser_prompts.source,
            "guesser_initial_prompt_path": guesser_prompts.initial_prompt_path,
            "guesser_turn_prompt_path": guesser_prompts.turn_prompt_path,
            "solved": False,
            "turns_used": len(turns),
            "final_question": outcome["final_question"],
            "final_question_correct": False,
            "run_dir": str(logger.root_dir),
            "error": str(exc),
            "error_type": error_type,
        }
