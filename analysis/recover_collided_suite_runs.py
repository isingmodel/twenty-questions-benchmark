from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twentyq.data import load_targets
from twentyq.run_single_target_suite import aggregate_results, load_suite_config, render_report

TARGETS = load_targets(ROOT / "data" / "all_target.csv")
OUTPUT_FILENAME = "guesser_session.json"


@dataclass
class GuesserEvent:
    index: int
    event: dict[str, Any]
    request_id: str
    previous_request_id: str | None
    root_request_id: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    return "".join(ch if ch.isalnum() else " " for ch in lowered).strip()


def _is_identity_question(question: str, target: dict[str, Any]) -> bool:
    normalized_question = f" {_normalize_text(question)} "
    for candidate in [target["name"], *target.get("aliases", [])]:
        normalized_candidate = _normalize_text(candidate)
        if normalized_candidate and f" {normalized_candidate} " in normalized_question:
            return True
    return False


def _request_id(event: dict[str, Any]) -> str:
    for key in ("request_id", "response_id", "interaction_id", "message_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"Missing request id in event: {event}")


def _previous_request_id(event: dict[str, Any]) -> str | None:
    for key in ("previous_request_id", "previous_response_id", "previous_interaction_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_question_from_judge_prompt(user_prompt: str) -> str:
    marker = "Current question:\n"
    end_marker = "\n\nReturn exactly one JSON object"
    start = user_prompt.find(marker)
    if start == -1:
        raise ValueError(f"Could not parse judge prompt question from: {user_prompt[:200]!r}")
    start += len(marker)
    end = user_prompt.find(end_marker, start)
    if end == -1:
        raise ValueError(f"Could not find end marker in judge prompt: {user_prompt[:200]!r}")
    return user_prompt[start:end]


def _build_guesser_chains(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[GuesserEvent]]]:
    guesser_events: list[GuesserEvent] = []
    by_request_id: dict[str, GuesserEvent] = {}

    for index, event in enumerate(events):
        if event.get("event") != "model_call" or event.get("role") != "guesser":
            continue
        request_id = _request_id(event)
        guesser_event = GuesserEvent(
            index=index,
            event=event,
            request_id=request_id,
            previous_request_id=_previous_request_id(event),
            root_request_id="",
        )
        guesser_events.append(guesser_event)
        by_request_id[request_id] = guesser_event

    root_cache: dict[str, str] = {}

    def resolve_root(request_id: str) -> str:
        cached = root_cache.get(request_id)
        if cached is not None:
            return cached
        current = by_request_id[request_id]
        if current.previous_request_id is None:
            root_cache[request_id] = request_id
            return request_id
        if current.previous_request_id not in by_request_id:
            raise ValueError(f"Missing parent request {current.previous_request_id!r} for {request_id!r}")
        root_request_id = resolve_root(current.previous_request_id)
        root_cache[request_id] = root_request_id
        return root_request_id

    chains: dict[str, list[GuesserEvent]] = defaultdict(list)
    for item in guesser_events:
        item.root_request_id = resolve_root(item.request_id)
        chains[item.root_request_id].append(item)

    for chain_events in chains.values():
        chain_events.sort(key=lambda item: int(item.event["turn"]))
    return guesser_events, chains


def _assign_judges(
    events: list[dict[str, Any]],
    guesser_events: list[GuesserEvent],
) -> dict[str, dict[str, Any]]:
    pending_by_question: dict[str, deque[GuesserEvent]] = defaultdict(deque)
    matched_judges: dict[str, dict[str, Any]] = {}
    guesser_by_index = {item.index: item for item in guesser_events}

    for index, event in enumerate(events):
        if index in guesser_by_index:
            question = str(guesser_by_index[index].event.get("raw_output", ""))
            pending_by_question[question].append(guesser_by_index[index])
            continue
        if event.get("event") != "model_call" or event.get("role") != "judge":
            continue
        question = _extract_question_from_judge_prompt(str(event.get("user_prompt", "")))
        queue = pending_by_question.get(question)
        if not queue:
            raise ValueError(f"Could not match judge event for question {question!r}")
        guesser_event = queue.popleft()
        matched_judges[guesser_event.request_id] = event

    for item in guesser_events:
        if item.request_id not in matched_judges:
            raise ValueError(f"Missing judge event for request {item.request_id!r}")
    return matched_judges


def _build_episode_turn(
    guesser_event: dict[str, Any],
    judge_event: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    question = str(guesser_event.get("raw_output", ""))
    return {
        "turn": int(guesser_event["turn"]),
        "question": question,
        "judgment": judge_event.get("normalized_output"),
        "judge_reason": judge_event.get("judge_reason"),
        "guesser_provider": guesser_event.get("provider"),
        "guesser_model": guesser_event.get("model"),
        "judge_provider": judge_event.get("provider"),
        "judge_model": judge_event.get("model"),
        "guesser_latency_ms": guesser_event.get("latency_ms"),
        "judge_latency_ms": judge_event.get("latency_ms"),
        "guesser_raw_output": question,
        "judge_raw_output": judge_event.get("raw_output"),
        "guesser_session_mode": guesser_event.get("session_mode"),
        "judge_session_mode": judge_event.get("session_mode"),
        "guesser_request_id": _request_id(guesser_event),
        "guesser_previous_request_id": _previous_request_id(guesser_event),
        "request_id": guesser_event.get("request_id"),
        "previous_request_id": guesser_event.get("previous_request_id"),
        "response_id": guesser_event.get("response_id"),
        "previous_response_id": guesser_event.get("previous_response_id"),
        "interaction_id": guesser_event.get("interaction_id"),
        "previous_interaction_id": guesser_event.get("previous_interaction_id"),
        "message_id": guesser_event.get("message_id"),
        "is_identity_question": _is_identity_question(question, target),
    }


def _build_episode(
    chain_events: list[GuesserEvent],
    matched_judges: dict[str, dict[str, Any]],
    target: dict[str, Any],
    summary: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    turns = [
        _build_episode_turn(item.event, matched_judges[item.request_id], target)
        for item in chain_events
    ]
    return {
        "mode": "full-game-test",
        "target_id": target["id"],
        "target_name": target["name"],
        "target_domain": target["domain"],
        "budget": budget,
        "turns": turns,
        "outcome": {
            "solved": summary["solved"],
            "turns_used": summary["turns_used"],
            "final_question": summary["final_question"],
            "final_question_correct": summary["final_question_correct"],
        },
    }


def _write_episode_markdown(path: Path, episode: dict[str, Any]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_run_config(
    run_id: str,
    run_dir: Path,
    run_started_event: dict[str, Any],
) -> dict[str, Any]:
    config = dict(run_started_event["config"])
    try:
        relative_run_dir = str(run_dir.relative_to(ROOT))
    except ValueError:
        relative_run_dir = str(run_dir)
    config["run_dir"] = relative_run_dir
    return {
        "run_id": run_id,
        "mode": run_started_event["mode"],
        "config": config,
    }


def _build_run_started_event(run_started_event: dict[str, Any], run_id: str) -> dict[str, Any]:
    recovered = dict(run_started_event)
    recovered["run_id"] = run_id
    return recovered


def _build_run_completed_event(
    ts: str,
    run_id: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ts": ts,
        "event": "run_completed",
        "run_id": run_id,
        "summary": dict(summary),
    }


def _build_guesser_session_payload(
    run_dir: Path,
    events_path: Path,
    chain_events: list[GuesserEvent],
    target_id: str,
    budget: int,
) -> dict[str, Any]:
    first_event = chain_events[0].event
    messages: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for item in chain_events:
        event = item.event
        request_id = _request_id(event)
        previous_request_id = _previous_request_id(event)
        user_prompt = str(event.get("user_prompt", ""))
        assistant_output = str(event.get("raw_output", ""))
        turns.append(
            {
                "turn": int(event["turn"]),
                "ts": event.get("ts"),
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
            }
        )
        messages.append(
            {
                "index": len(messages) + 1,
                "role": "user",
                "turn": int(event["turn"]),
                "ts": event.get("ts"),
                "request_id": request_id,
                "previous_request_id": previous_request_id,
                "text": user_prompt,
            }
        )
        messages.append(
            {
                "index": len(messages) + 1,
                "role": "assistant",
                "turn": int(event["turn"]),
                "ts": event.get("ts"),
                "request_id": request_id,
                "previous_request_id": previous_request_id,
                "text": assistant_output,
            }
        )
    return {
        "format": "guesser-session-v1",
        "source_events_path": str(events_path),
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "provider": first_event.get("provider"),
        "model": first_event.get("model"),
        "target_id": target_id,
        "budget": budget,
        "session_mode": first_event.get("session_mode"),
        "turn_count": len(turns),
        "message_count": len(messages),
        "turns": turns,
        "messages": messages,
    }


def _chain_signature(chain_events: list[GuesserEvent]) -> tuple[int, str]:
    return len(chain_events), str(chain_events[-1].event.get("raw_output", ""))


def _summary_signature(summary: dict[str, Any]) -> tuple[int, str]:
    return int(summary["turns_used"]), str(summary["final_question"])


def _backup_original_folder(run_dir: Path) -> Path:
    backup_dir = run_dir / "recovery_backup_pre_split"
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in ["events.jsonl", "summary.json", "run_config.json", OUTPUT_FILENAME]:
        path = run_dir / name
        if path.exists():
            shutil.copy2(path, backup_dir / name)
    episodes_dir = run_dir / "episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, backup_dir / "episodes")
    return backup_dir


def _next_run_index(runs_dir: Path) -> int:
    highest = 0
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("run-"):
            continue
        try:
            highest = max(highest, int(name[4:8]))
        except ValueError:
            continue
    return highest + 1


def _render_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    run_started_event: dict[str, Any],
    run_completed_ts: str,
    episode: dict[str, Any],
    summary: dict[str, Any],
    chain_events: list[GuesserEvent],
    matched_judges: dict[str, dict[str, Any]],
    backup_original: bool,
) -> None:
    if backup_original:
        _backup_original_folder(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rebuilt_events = [_build_run_started_event(run_started_event, run_id)]
    for item in chain_events:
        guesser_event = dict(item.event)
        guesser_event["run_id"] = run_id
        rebuilt_events.append(guesser_event)
        rebuilt_events.append(dict(matched_judges[item.request_id]))
    rebuilt_events.append(_build_run_completed_event(run_completed_ts, run_id, summary))

    _write_events(run_dir / "events.jsonl", rebuilt_events)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "run_config.json", _build_run_config(run_id, run_dir, run_started_event))
    _write_json(run_dir / "episodes" / f"{episode['target_id']}.json", episode)
    _write_episode_markdown(run_dir / "episodes" / f"{episode['target_id']}.md", episode)
    _write_json(
        run_dir / OUTPUT_FILENAME,
        _build_guesser_session_payload(
            run_dir=run_dir,
            events_path=run_dir / "events.jsonl",
            chain_events=chain_events,
            target_id=episode["target_id"],
            budget=int(episode["budget"]),
        ),
    )


def recover_suite(suite_dir: Path, config_path: Path, apply: bool) -> dict[str, Any]:
    results_wrapper = _read_json(suite_dir / "results.json")
    results = list(results_wrapper["results"])
    rows_by_run_dir: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(results):
        rows_by_run_dir[str(row["run_dir"])].append((index, row))

    duplicate_groups = {run_dir: items for run_dir, items in rows_by_run_dir.items() if len(items) > 1}
    plan: list[dict[str, Any]] = []
    next_run_index = _next_run_index(suite_dir / "runs")

    for run_dir_str, duplicate_rows in sorted(duplicate_groups.items()):
        run_dir = Path(run_dir_str)
        events = _read_events(run_dir / "events.jsonl")
        run_started_events = [event for event in events if event.get("event") == "run_started"]
        run_completed_events = [event for event in events if event.get("event") == "run_completed"]
        guesser_events, chains = _build_guesser_chains(events)
        matched_judges = _assign_judges(events, guesser_events)
        if len(chains) != len(duplicate_rows):
            raise ValueError(f"{run_dir} has {len(chains)} chains but {len(duplicate_rows)} duplicate rows")

        current_summary = _read_json(run_dir / "summary.json")
        target = TARGETS[current_summary["target_id"]]
        chain_records: list[dict[str, Any]] = []
        unmatched_rows = list(duplicate_rows)
        for root_request_id, chain_events in sorted(
            chains.items(),
            key=lambda item: item[1][0].index,
        ):
            signature = _chain_signature(chain_events)
            matched_index = None
            for row_index, (_, row) in enumerate(unmatched_rows):
                if _summary_signature(row) == signature:
                    matched_index = row_index
                    break
            if matched_index is None:
                raise ValueError(f"Could not match chain {root_request_id} in {run_dir}")
            row_position, row = unmatched_rows.pop(matched_index)
            summary = {
                "run_id": row["run_id"],
                "mode": row["mode"],
                "target_id": row["target_id"],
                "target_name": row["target_name"],
                "solved": row["solved"],
                "turns_used": row["turns_used"],
                "final_question": row["final_question"],
                "final_question_correct": row["final_question_correct"],
                "run_dir": row["run_dir"],
            }
            episode = _build_episode(
                chain_events=chain_events,
                matched_judges=matched_judges,
                target=target,
                summary=summary,
                budget=int(run_started_events[0]["config"]["budget"]),
            )
            chain_records.append(
                {
                    "root_request_id": root_request_id,
                    "chain_events": chain_events,
                    "matched_judges": matched_judges,
                    "row_index": row_position,
                    "row": row,
                    "summary": summary,
                    "episode": episode,
                }
            )

        keep_signature = _summary_signature(current_summary)
        keep_record = next(record for record in chain_records if _summary_signature(record["summary"]) == keep_signature)
        recovered_records = [record for record in chain_records if record is not keep_record]

        assignments: list[dict[str, Any]] = []
        assignments.append(
            {
                "target_run_dir": run_dir,
                "target_run_id": run_dir.name,
                "update_results": False,
                "record": keep_record,
                "backup_original": True,
            }
        )
        for record in recovered_records:
            new_run_id = f"run-{next_run_index:04d}__{run_dir.name.split('__', 1)[1]}"
            next_run_index += 1
            new_run_dir = run_dir.parent / new_run_id
            assignments.append(
                {
                    "target_run_dir": new_run_dir,
                    "target_run_id": new_run_id,
                    "update_results": True,
                    "record": record,
                    "backup_original": False,
                }
            )
        plan.append(
            {
                "original_run_dir": str(run_dir),
                "current_summary_signature": keep_signature,
                "assignments": [
                    {
                        "target_run_dir": str(item["target_run_dir"]),
                        "target_run_id": item["target_run_id"],
                        "turns_used": item["record"]["summary"]["turns_used"],
                        "final_question": item["record"]["summary"]["final_question"],
                        "repetition_index": item["record"]["row"]["repetition_index"],
                        "update_results": item["update_results"],
                    }
                    for item in assignments
                ],
            }
        )

        if not apply:
            continue

        for assignment_index, assignment in enumerate(assignments):
            target_run_dir = assignment["target_run_dir"]
            target_run_id = assignment["target_run_id"]
            row = assignment["record"]["row"]
            summary = dict(assignment["record"]["summary"])
            summary["run_id"] = target_run_id
            summary["run_dir"] = str(target_run_dir)
            episode = dict(assignment["record"]["episode"])
            episode["outcome"] = dict(episode["outcome"])
            run_started_event = run_started_events[min(assignment_index, len(run_started_events) - 1)]
            matching_completed = next(
                event
                for event in run_completed_events
                if _summary_signature(event["summary"]) == _summary_signature(assignment["record"]["summary"])
            )

            _render_artifacts(
                run_dir=target_run_dir,
                run_id=target_run_id,
                run_started_event=run_started_event,
                run_completed_ts=str(matching_completed["ts"]),
                episode=episode,
                summary=summary,
                chain_events=assignment["record"]["chain_events"],
                matched_judges=assignment["record"]["matched_judges"],
                backup_original=assignment["backup_original"],
            )

            results[assignment["record"]["row_index"]]["run_id"] = target_run_id
            results[assignment["record"]["row_index"]]["run_dir"] = str(target_run_dir)

    if apply and duplicate_groups:
        _write_json(suite_dir / "results.json", {"results": results})
        config = load_suite_config(config_path)
        aggregate = aggregate_results(config, results, suite_dir)
        _write_json(suite_dir / "aggregate.json", aggregate)
        (suite_dir / "report.md").write_text(render_report(config, aggregate), encoding="utf-8")
        status_path = suite_dir / "status.json"
        status = _read_json(status_path)
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        status["updated_at"] = now
        _write_json(status_path, status)

    return {
        "suite_dir": str(suite_dir),
        "duplicate_group_count": len(duplicate_groups),
        "plan": plan,
        "applied": apply,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover collided suite runs caused by duplicate run folders.")
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply the recovery plan.")
    args = parser.parse_args()

    result = recover_suite(args.suite_dir.resolve(), args.config.resolve(), apply=args.apply)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
