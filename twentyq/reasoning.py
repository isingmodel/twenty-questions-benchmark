from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningConfig:
    thinking_level: str | None = None
    thinking_budget: int | None = None
    reasoning_effort: str | None = None
