from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


KNOWN_DOMAIN_PREFIXES = {
    "animal": "animals",
    "character": "characters",
    "food": "foods",
    "object": "objects",
    "person": "people",
    "place": "places",
}


def _require_non_empty_string(record: dict[str, Any], field_name: str, source: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string field {field_name!r} in {source}")
    return value.strip()


def _normalize_delimited_list(value: Any, field_name: str, source: str) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"Expected string field {field_name!r} in {source}")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value.split("|"):
        cleaned = item.strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    if not normalized:
        raise ValueError(f"Expected at least one value in {field_name!r} for {source}")
    return normalized


def _normalize_domain(raw_domain: Any, target_id: str, source: str) -> str:
    if isinstance(raw_domain, str) and raw_domain.strip():
        return raw_domain.strip()

    prefix = target_id.split("_", 1)[0]
    inferred = KNOWN_DOMAIN_PREFIXES.get(prefix)
    if inferred:
        return inferred
    raise ValueError(f"Could not infer domain for target {target_id!r} in {source}")


def _normalize_target_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    target_id = _require_non_empty_string(record, "id", source)
    name = _require_non_empty_string(record, "name", source)
    description = _require_non_empty_string(record, "description", source)
    aliases = _normalize_delimited_list(record.get("aliases"), "aliases", source)
    if name not in aliases:
        aliases = [name, *aliases]

    normalized = dict(record)
    normalized.update(
        {
            "id": target_id,
            "name": name,
            "domain": _normalize_domain(record.get("domain"), target_id, source),
            "aliases": aliases,
            "description": description,
        }
    )
    return normalized


def load_targets(targets_dir: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for path in sorted(targets_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Missing CSV header in {path}")
            required = {"id", "name", "domain", "aliases", "description"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Missing required CSV columns {sorted(missing)!r} in {path}")
            for line_number, raw_record in enumerate(reader, start=2):
                source = f"{path}:{line_number}"
                record = _normalize_target_record(raw_record, source)
                target_id = record["id"]
                if target_id in targets:
                    raise ValueError(f"Duplicate target id {target_id!r} in {source}")
                targets[target_id] = record
    return targets


def load_split(split_path: Path, targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()
    for line_number, raw_line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), start=1):
        target_id = raw_line.strip()
        if not target_id:
            continue
        if target_id not in targets:
            raise ValueError(f"Unknown target id {target_id!r} in {split_path}:{line_number}")
        if target_id in seen_target_ids:
            raise ValueError(f"Duplicate target id {target_id!r} in {split_path}:{line_number}")
        seen_target_ids.add(target_id)
        episodes.append(targets[target_id])
    return episodes
