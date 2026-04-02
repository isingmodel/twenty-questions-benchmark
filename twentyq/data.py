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


def _resolve_data_path(data_path: Path) -> Path:
    if data_path.is_file():
        return data_path
    candidates = [
        data_path / "targets.csv",
        data_path / "all_target.csv",
        data_path.parent / "all_target.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find targets CSV for {data_path}")


def load_data(data_path: Path) -> list[dict[str, Any]]:
    data_path = _resolve_data_path(data_path)
    targets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {data_path}")
        required = {"id", "name", "domain", "aliases", "description"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required CSV columns {sorted(missing)!r} in {data_path}")
        for line_number, raw_record in enumerate(reader, start=2):
            source = f"{data_path}:{line_number}"
            record = _normalize_target_record(raw_record, source)
            target_id = record["id"]
            if target_id in seen_ids:
                raise ValueError(f"Duplicate target id {target_id!r} in {source}")
            seen_ids.add(target_id)
            targets.append(record)
    return targets


def load_targets(data_path: Path) -> dict[str, dict[str, Any]]:
    return {target["id"]: target for target in load_data(data_path)}
