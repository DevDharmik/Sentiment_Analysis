"""Utilities for reading Xquik export files."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from typing import Any

TEXT_FIELDS = ("comment", "tweet", "text", "full_text", "tweet_text", "content", "body")
LIKE_FIELDS = ("likes", "like_count", "favorite_count", "favorites")
DATE_FIELDS = ("published", "created_at", "date", "timestamp")
AUTHOR_FIELDS = ("author", "username", "user", "screen_name")


class ExportFormatError(ValueError):
    """Raised when an export cannot be decoded or parsed."""


def _iter_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("data", "results", "comments", "tweets", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                yield from _iter_records(nested)
                return
        yield value
        return

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield from _iter_records(item)


def _read_json_or_jsonl(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line in raw.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            rows.extend(_iter_records(json.loads(candidate)))
        return rows
    return list(_iter_records(parsed))


def _read_csv(raw: str) -> list[dict[str, Any]]:
    return list(csv.DictReader(io.StringIO(raw)))


def _first_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_int(row: dict[str, Any], fields: tuple[str, ...]) -> int:
    raw = _first_text(row, fields)
    if not raw:
        return 0
    try:
        return int(float(raw))
    except ValueError:
        return 0


def load_xquik_rows(payload: bytes) -> list[dict[str, object]]:
    """Return normalized comment rows from JSON, JSONL, or CSV export bytes."""
    try:
        raw = payload.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise ExportFormatError("Export must use UTF-8 encoding.") from error
    if not raw:
        return []

    try:
        records = _read_json_or_jsonl(raw) if raw[:1] in "[{" else _read_csv(raw)
    except (csv.Error, json.JSONDecodeError) as error:
        raise ExportFormatError("Export is not valid JSON, JSONL, or CSV.") from error

    rows: list[dict[str, object]] = []
    for record in records:
        text = _first_text(record, TEXT_FIELDS)
        if text:
            rows.append(
                {
                    "author": _first_text(record, AUTHOR_FIELDS) or "Xquik export",
                    "text": text,
                    "likes": _first_int(record, LIKE_FIELDS),
                    "published": _first_text(record, DATE_FIELDS),
                }
            )
    return rows
