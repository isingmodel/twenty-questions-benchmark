from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ELIGIBLE_PROVIDERS = {"gemini", "openai"}
OUTPUT_FILENAME = "guesser_session.json"


def _load_events(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{events_path}:{line_number} is not a JSON object")
        payload["_source_line"] = line_number
        events.append(payload)
    return events


def _get_run_started_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") == "run_started":
            return event
    return None


def _get_guesser_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guesser_events = [
        event
        for event in events
        if event.get("event") == "model_call" and event.get("role") == "guesser"
    ]
    guesser_events.sort(key=lambda event: int(event.get("turn", 0)))
    return guesser_events


def _resolve_provider(guesser_events: list[dict[str, Any]], run_started: dict[str, Any] | None) -> str | None:
    if guesser_events:
        provider = guesser_events[0].get("provider")
        if isinstance(provider, str) and provider:
            return provider
    if run_started is None:
        return None
    config = run_started.get("config")
    if not isinstance(config, dict):
        return None
    provider = config.get("guesser_provider")
    if isinstance(provider, str) and provider:
        return provider
    return None


def _request_id(event: dict[str, Any]) -> str | None:
    for key in ("request_id", "response_id", "interaction_id", "message_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _previous_request_id(event: dict[str, Any]) -> str | None:
    for key in (
        "previous_request_id",
        "previous_response_id",
        "previous_interaction_id",
    ):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _build_session_payload(events_path: Path) -> dict[str, Any] | None:
    events = _load_events(events_path)
    run_started = _get_run_started_event(events)
    guesser_events = _get_guesser_events(events)
    provider = _resolve_provider(guesser_events, run_started)
    if provider not in ELIGIBLE_PROVIDERS:
        return None

    run_config = {}
    run_id = events_path.parent.name
    if run_started is not None:
        if isinstance(run_started.get("run_id"), str) and run_started["run_id"]:
            run_id = run_started["run_id"]
        if isinstance(run_started.get("config"), dict):
            run_config = dict(run_started["config"])

    turns: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []

    for event in guesser_events:
        turn_number = int(event.get("turn", 0))
        request_id = _request_id(event)
        previous_request_id = _previous_request_id(event)
        timestamp = event.get("ts")
        user_prompt = str(event.get("user_prompt", ""))
        assistant_output = str(event.get("raw_output", ""))

        turn_payload = {
            "turn": turn_number,
            "ts": timestamp,
            "request_id": request_id,
            "previous_request_id": previous_request_id,
            "provider": event.get("provider"),
            "model": event.get("model"),
            "session_mode": event.get("session_mode"),
            "generation_config": event.get("generation_config"),
            "system_prompt": event.get("system_prompt"),
            "turn_prompt": event.get("turn_prompt"),
            "user_prompt": user_prompt,
            "assistant_output": assistant_output,
            "source_event_line": event.get("_source_line"),
        }
        turns.append(turn_payload)

        messages.append(
            {
                "index": len(messages) + 1,
                "role": "user",
                "turn": turn_number,
                "ts": timestamp,
                "request_id": request_id,
                "previous_request_id": previous_request_id,
                "text": user_prompt,
            }
        )
        messages.append(
            {
                "index": len(messages) + 1,
                "role": "assistant",
                "turn": turn_number,
                "ts": timestamp,
                "request_id": request_id,
                "previous_request_id": previous_request_id,
                "text": assistant_output,
            }
        )

    return {
        "format": "guesser-session-v1",
        "source_events_path": str(events_path),
        "run_dir": str(events_path.parent),
        "run_id": run_id,
        "provider": provider,
        "model": run_config.get("guesser_model"),
        "target_id": run_config.get("target_id"),
        "budget": run_config.get("budget"),
        "session_mode": guesser_events[0].get("session_mode") if guesser_events else None,
        "turn_count": len(turns),
        "message_count": len(messages),
        "turns": turns,
        "messages": messages,
    }


def _iter_events_paths(reports_root: Path) -> list[Path]:
    paths = [
        path
        for path in reports_root.rglob("events.jsonl")
        if path.parent.name.startswith("run-") and path.parent.parent.name == "runs"
    ]
    return sorted(paths)


def reconstruct_sessions(reports_root: Path, overwrite: bool = True) -> tuple[int, int]:
    written = 0
    skipped = 0
    for events_path in _iter_events_paths(reports_root):
        session_payload = _build_session_payload(events_path)
        if session_payload is None:
            skipped += 1
            continue

        output_path = events_path.parent / OUTPUT_FILENAME
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        output_path.write_text(
            json.dumps(session_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct guesser session transcripts for OpenAI and Gemini runs."
    )
    parser.add_argument(
        "--reports-root",
        default="reports",
        help="Path to the reports directory. Defaults to ./reports",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Leave existing guesser_session.json files untouched.",
    )
    args = parser.parse_args()

    reports_root = Path(args.reports_root).resolve()
    written, skipped = reconstruct_sessions(reports_root, overwrite=not args.no_overwrite)
    print(
        json.dumps(
            {
                "reports_root": str(reports_root),
                "written": written,
                "skipped": skipped,
                "output_filename": OUTPUT_FILENAME,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
