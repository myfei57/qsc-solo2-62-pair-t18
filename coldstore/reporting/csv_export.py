"""CSV rendering for operator exports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from coldstore.audit import Auditor
    from coldstore.console.snapshot import Snapshot
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


def snapshot_export(snapshot: Snapshot) -> tuple[list[str], list[list[object]]]:
    """Columns and rows for one consistent plant snapshot.

    Every row repeats the snapshot identity so the exported file stays
    traceable to the exact document the console served.
    """

    columns = ["snapshot_id", "version", "generated_at", "segment", "available", "error", "payload"]
    rows = [
        [
            snapshot.snapshot_id,
            snapshot.tag,
            snapshot.generated_at,
            name,
            snapshot.availability[name],
            snapshot.errors.get(name, ""),
            "" if segment is None else json.dumps(segment, ensure_ascii=False, sort_keys=True),
        ]
        for name, segment in snapshot.segments.items()
    ]
    return columns, rows
