from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .clients import (
    GeminiClient,
    GeminiInteractionSession,
    HTTPError,
    OpenAIClient,
    OpenAIResponsesSession,
)
from .env import get_required_env
from .prompts import ROOT, load_prompt, render_template
from .reasoning import ReasoningConfig
from .run_logs import RunLogger


PROVIDER = "gemini"
OPENAI_PROVIDER = "openai"
THINKING_LEVEL_CHOICES = ("minimal", "low", "medium", "high")
REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high")
DEFAULT_TARGET_ID = "object_toothbrush"
DEFAULT_BUDGET = 20
DEFAULT_GUESSER_MODEL = "gemini-2.5-flash"
DEFAULT_JUDGE_MODEL = "gemini-3-flash-preview"
JUDGMENT_TRUE = "Yes"
JUDGMENT_FALSE = "No"
JUDGMENT_AMBIGUOUS = "Ambiguous"
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


@dataclass(frozen=True)
class JudgeDecision:
    label: str
    reason: str | None = None


@dataclass
class FullGameConfig:
    target_id: str
    budget: int
    guesser_model: str
    judge_model: str
    guesser_reasoning: ReasoningConfig
    judge_reasoning: ReasoningConfig
    run_dir: Path | None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def provider_for_model(model: str) -> str:
    normalized = model.strip().lower()
    if normalized.startswith("gemini-"):
        return PROVIDER
    if normalized.startswith(("gpt-", "o1", "o3", "o4", "codex-")):
        return OPENAI_PROVIDER
    raise ValueError(f"Unsupported model provider for model {model!r}")


def _create_gemini_client() -> GeminiClient:
    return GeminiClient(get_required_env("gemini_key"))


def _create_openai_client() -> OpenAIClient:
    try:
        api_key = get_required_env("OPENAI_API_KEY")
    except RuntimeError:
        api_key = get_required_env("openai_key")
    return OpenAIClient(api_key)


def _create_client_for_model(model: str) -> Any:
    provider = provider_for_model(model)
    if provider == PROVIDER:
        return _create_gemini_client()
    return _create_openai_client()


def _create_guesser_session(client: Any, model: str, system_prompt: str, initial_user_prompt: str) -> Any:
    provider = provider_for_model(model)
    if provider == PROVIDER:
        return GeminiInteractionSession(
            client=client,
            model=model,
            system_prompt=system_prompt,
            initial_user_prompt=initial_user_prompt,
        )
    return OpenAIResponsesSession(
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


def _normalize_text(text: str) -> str:
    value = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    value = re.sub(r"\s+", " ", value).strip()
    for prefix in ("a ", "an ", "the "):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.strip()


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


def _parse_judge_response(raw_output: str) -> JudgeDecision:
    raw_text = str(raw_output).strip()
    if not raw_text:
        raise ValueError("Judge returned empty output")

    label_source = raw_text
    reason: str | None = None
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

    return JudgeDecision(label=_normalize_judge_output(label_source), reason=reason)


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


def _is_identity_question(question: str, target: dict[str, Any]) -> bool:
    normalized_question = f" {_normalize_text(question)} "
    for candidate in [target["name"], *target.get("aliases", [])]:
        normalized_candidate = _normalize_text(candidate)
        if normalized_candidate and f" {normalized_candidate} " in normalized_question:
            return True
    return False


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

    normalized_model = model.strip().lower()
    if provider_for_model(model) == OPENAI_PROVIDER:
        if thinking_budget is not None:
            raise ValueError(f"{role} model {model!r} does not support thinking budgets")
        if thinking_level is None:
            return ReasoningConfig()
        if thinking_level not in THINKING_LEVEL_CHOICES:
            raise ValueError(f"{role} model {model!r} has unsupported thinking level {thinking_level!r}")
        return ReasoningConfig(reasoning_effort=thinking_level)
    if normalized_model.startswith("gemini-3") and thinking_budget is not None:
        raise ValueError(f"{role} model {model!r} uses thinking levels, not thinking budgets")
    if normalized_model.startswith("gemini-3-pro") and thinking_level not in {None, "low", "high"}:
        raise ValueError(f"{role} model {model!r} only supports thinking levels 'low' and 'high'")
    if normalized_model.startswith("gemini-3.1-pro") and thinking_level not in {None, "low", "medium", "high"}:
        raise ValueError(f"{role} model {model!r} only supports thinking levels 'low', 'medium', and 'high'")
    if (
        normalized_model.startswith("gemini-3-flash")
        or normalized_model.startswith("gemini-3.1-flash-lite")
    ) and thinking_level not in {None, *THINKING_LEVEL_CHOICES}:
        raise ValueError(f"{role} model {model!r} has unsupported thinking level {thinking_level!r}")
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
    if provider_for_model(model) == OPENAI_PROVIDER:
        if effort is None:
            return ReasoningConfig()
        normalized_effort = effort.strip().lower()
        if normalized_effort not in REASONING_EFFORT_CHOICES:
            raise ValueError(f"Unsupported reasoning effort {effort!r}")
        return ReasoningConfig(reasoning_effort=normalized_effort)
    if effort is None:
        return ReasoningConfig()

    normalized_effort = effort.strip().lower()
    if normalized_effort not in REASONING_EFFORT_CHOICES:
        raise ValueError(f"Unsupported reasoning effort {effort!r}")

    normalized_model = model.strip().lower()
    if normalized_model.startswith("gemini-2.5-flash-lite"):
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=GEMINI_25_FLASH_THINKING_LEVEL_BY_EFFORT[normalized_effort],
            thinking_budget=None,
        )
    if normalized_model.startswith("gemini-2.5-flash"):
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=GEMINI_25_FLASH_THINKING_LEVEL_BY_EFFORT[normalized_effort],
            thinking_budget=None,
        )
    if normalized_model.startswith("gemini-2.5-pro"):
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=GEMINI_25_PRO_THINKING_LEVEL_BY_EFFORT[normalized_effort],
            thinking_budget=None,
        )
    if normalized_model.startswith("gemini-3.1-flash-lite"):
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=GEMINI_3_FLASH_THINKING_LEVEL_BY_EFFORT[normalized_effort],
            thinking_budget=None,
        )
    if normalized_model.startswith("gemini-3-flash"):
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=GEMINI_3_FLASH_THINKING_LEVEL_BY_EFFORT[normalized_effort],
            thinking_budget=None,
        )
    if normalized_model.startswith("gemini-3.1-pro"):
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=GEMINI_31_PRO_THINKING_LEVEL_BY_EFFORT[normalized_effort],
            thinking_budget=None,
        )
    if normalized_model.startswith("gemini-3-pro"):
        return _validate_reasoning_config(
            role=role,
            model=model,
            thinking_level=GEMINI_3_PRO_THINKING_LEVEL_BY_EFFORT[normalized_effort],
            thinking_budget=None,
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
    guesser_initial = load_prompt("guesser-initial.txt")
    guesser_turn_template = load_prompt("guesser-turn.txt")
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
                guesser_interaction_id,
                guesser_previous_interaction_id,
            ) = guesser_result
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
                    "user_prompt": guesser_user_prompt,
                    "turn_prompt": turn_prompt,
                    "raw_output": guesser_raw_output,
                    "generation_config": _reasoning_to_payload(config.guesser_reasoning),
                    "interaction_id": guesser_interaction_id,
                    "previous_interaction_id": guesser_previous_interaction_id,
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
            judge_decision = _parse_judge_response(judge_raw_output)
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
                    "generation_config": _reasoning_to_payload(config.judge_reasoning),
                }
            )

            is_identity_question = _is_identity_question(guesser_raw_output, target)
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
                    "guesser_session_mode": guesser_session.session_mode,
                    "judge_session_mode": judge_session_mode,
                    "guesser_interaction_id": guesser_interaction_id,
                    "guesser_previous_interaction_id": guesser_previous_interaction_id,
                    "is_identity_question": is_identity_question,
                }
            )
            outcome["turns_used"] = turn_number
            outcome["final_question"] = guesser_raw_output
            outcome["final_question_correct"] = bool(is_identity_question and judgment == JUDGMENT_TRUE)
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
            "solved": False,
            "turns_used": len(turns),
            "final_question": outcome["final_question"],
            "final_question_correct": False,
            "run_dir": str(logger.root_dir),
            "error": str(exc),
            "error_type": error_type,
        }
