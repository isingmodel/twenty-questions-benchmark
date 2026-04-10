from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
DEFAULT_GUESSER_PROMPT_SET = "default"


@dataclass(frozen=True)
class GuesserPromptSet:
    name: str
    initial_prompt_name: str
    turn_prompt_name: str


@dataclass(frozen=True)
class LoadedGuesserPrompts:
    name: str
    source: str
    initial_prompt_path: str
    turn_prompt_path: str
    initial_prompt: str
    turn_prompt: str


BUILTIN_GUESSER_PROMPT_SETS = {
    "default": GuesserPromptSet(
        name="default",
        initial_prompt_name="guesser-initial.txt",
        turn_prompt_name="guesser-turn.txt",
    ),
    "strategic": GuesserPromptSet(
        name="strategic",
        initial_prompt_name="guesser-strategic-initial.txt",
        turn_prompt_name="guesser-turn.txt",
    ),
}


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def render_template(template: str, **kwargs: str) -> str:
    rendered = template
    for key, value in kwargs.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve_builtin_guesser_prompt_set(name: str | None = None) -> GuesserPromptSet:
    normalized = (name or DEFAULT_GUESSER_PROMPT_SET).strip().lower()
    prompt_set = BUILTIN_GUESSER_PROMPT_SETS.get(normalized)
    if prompt_set is None:
        supported = ", ".join(sorted(BUILTIN_GUESSER_PROMPT_SETS))
        raise ValueError(
            f"Unknown built-in guesser prompt set {name!r}. Supported values: {supported}"
        )
    return prompt_set


def load_guesser_prompts(
    *,
    prompt_set: str | None = None,
    initial_prompt_path: Path | None = None,
    turn_prompt_path: Path | None = None,
) -> LoadedGuesserPrompts:
    if (initial_prompt_path is None) != (turn_prompt_path is None):
        raise ValueError(
            "Custom guesser prompts require both initial_prompt_path and turn_prompt_path"
        )

    if initial_prompt_path is not None and turn_prompt_path is not None:
        normalized_name = (prompt_set or "custom").strip() or "custom"
        initial_path = initial_prompt_path.resolve()
        turn_path = turn_prompt_path.resolve()
        return LoadedGuesserPrompts(
            name=normalized_name,
            source="custom",
            initial_prompt_path=_display_path(initial_path),
            turn_prompt_path=_display_path(turn_path),
            initial_prompt=initial_path.read_text(encoding="utf-8").strip(),
            turn_prompt=turn_path.read_text(encoding="utf-8").strip(),
        )

    builtin = resolve_builtin_guesser_prompt_set(prompt_set)
    initial_path = PROMPTS_DIR / builtin.initial_prompt_name
    turn_path = PROMPTS_DIR / builtin.turn_prompt_name
    return LoadedGuesserPrompts(
        name=builtin.name,
        source="builtin",
        initial_prompt_path=_display_path(initial_path),
        turn_prompt_path=_display_path(turn_path),
        initial_prompt=initial_path.read_text(encoding="utf-8").strip(),
        turn_prompt=turn_path.read_text(encoding="utf-8").strip(),
    )

