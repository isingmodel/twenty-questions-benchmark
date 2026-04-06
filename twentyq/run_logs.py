from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any


_RUN_ID_LOCK = Lock()


def _sanitize_fragment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def _next_run_index(runs_dir: Path) -> int:
    highest = 0
    if not runs_dir.exists():
        return 1
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"run-(\d{4})__", path.name)
        if not match:
            continue
        highest = max(highest, int(match.group(1)))
    return highest + 1


class RunLogger:
    def __init__(self, root_dir: Path, run_id: str) -> None:
        self.root_dir = root_dir
        self.run_id = run_id
        self.events_path = self.root_dir / "events.jsonl"
        self.episodes_dir = self.root_dir / "episodes"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, runs_dir: Path, split: str, guesser_provider: str, guesser_model: str) -> "RunLogger":
        slug = "__".join(
            [
                _sanitize_fragment(split),
                _sanitize_fragment(guesser_provider),
                _sanitize_fragment(guesser_model),
            ]
        )
        with _RUN_ID_LOCK:
            while True:
                run_index = _next_run_index(runs_dir)
                run_id = f"run-{run_index:04d}__{slug}"
                root_dir = runs_dir / run_id
                try:
                    root_dir.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    continue
                return cls(root_dir, run_id)

    def _json_default(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> None:
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=self._json_default) + "\n", encoding="utf-8")

    def log_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, default=self._json_default) + "\n")

    def write_episode_artifacts(self, episode: dict[str, Any]) -> None:
        target_id = episode["target_id"]
        self.write_json(f"episodes/{target_id}.json", episode)

        metric_lines: list[str] = []
        for key, value in episode.get("metrics", {}).items():
            metric_lines.append(f"- {key}: {value}")

        lines = [
            f"# Episode {target_id}",
            "",
            f"- target_name: {episode['target_name']}",
            f"- target_domain: {episode['target_domain']}",
            f"- budget: {episode['budget']}",
            *metric_lines,
            "",
            "## Turns",
            "",
        ]
        for turn in episode["turns"]:
            guesser_provider = turn.get("guesser_provider", "unknown")
            judge_provider = turn.get("judge_provider", "unknown")
            lines.extend(
                [
                    f"### Turn {turn['turn']}",
                    "",
                    f"Question: {turn['question']}",
                    f"Judgment: {turn['judgment']}",
                    f"Judge reason: {turn.get('judge_reason')}",
                    f"Guesser provider/model: {guesser_provider} / {turn['guesser_model']}",
                    f"Judge provider/model: {judge_provider} / {turn['judge_model']}",
                    f"Guesser session mode: {turn.get('guesser_session_mode')}",
                    f"Judge session mode: {turn.get('judge_session_mode')}",
                    f"Guesser latency ms: {turn['guesser_latency_ms']}",
                    f"Judge latency ms: {turn['judge_latency_ms']}",
                    "",
                    "Guesser raw output:",
                    "```text",
                    turn["guesser_raw_output"],
                    "```",
                    "",
                    "Judge raw output:",
                    "```text",
                    turn["judge_raw_output"],
                    "```",
                    "",
                ]
            )
        (self.episodes_dir / f"{target_id}.md").write_text("\n".join(lines), encoding="utf-8")
