"""CSV rendering for operator exports."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from coldstore.audit import Auditor
    from coldstore.console.coordinator import SnapshotDocument
    from coldstore.console.telemetry import Telemetry

QUOTE_TRIGGERS = (",", '"', "\n")


def _escape(value: object) -> str:
    text = "" if value is None else str(value)
    if any(trigger in text for trigger in QUOTE_TRIGGERS):
        return '"' + text.replace('"', '""') + '"'
    return text


def to_csv(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a header row and body rows as comma separated text."""

    lines = [",".join(_escape(column) for column in columns)]
    for row in rows:
        lines.append(",".join(_escape(value) for value in row))
    return "\n".join(lines) + "\n"


def audit_export(auditor: Auditor) -> tuple[list[str], list[list[str]]]:
    """Columns and rows for the runtime audit trail."""

    columns = ["time", "kind", "detail", "id"]
    rows = [[entry.time, entry.kind, entry.detail, entry.id] for entry in auditor.entries()]
    return columns, rows


def telemetry_export(telemetry: Telemetry) -> tuple[list[str], list[list[object]]]:
    """Columns and rows for the telemetry counters."""

    return ["metric", "value"], [[key, value] for key, value in telemetry.as_dict().items()]


def _flatten(prefix: str, value: object) -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        if not value:
            yield prefix, "{}"
        for key in sorted(value):
            yield from _flatten(f"{prefix}.{key}", value[key])
    elif isinstance(value, list):
        yield prefix, json.dumps(value, ensure_ascii=False)
    else:
        yield prefix, value


def snapshot_export(document: SnapshotDocument) -> tuple[list[str], list[list[object]]]:
    """Columns and rows for one cached consistent snapshot document."""

    rows: list[list[object]] = [
        ["meta", "version", document.version],
        ["meta", "generated_at", document.generated_at],
        ["meta", "partial", document.partial],
    ]
    for name, state in document.availability.items():
        marker = "available" if state.available else f"unavailable: {state.error}"
        rows.append(["availability", name, marker])
    for name, line in document.describe.items():
        rows.append(["describe", name, line])
    for name, payload in document.segments.items():
        if payload is None:
            continue
        for key, value in _flatten(f"segments.{name}", payload):
            rows.append(["segment", key, value])
    return ["section", "key", "value"], rows
