"""Consistent plant snapshot exposed by the console.

The old collector walked every component one read at a time, so a snapshot
taken while the plant was being driven could mix room temperatures from one
moment with compressor stages from another. This module replaces that walk
with a versioned, cached, partially available snapshot.

Decisions, pinned down:

* **Frozen version.** Every plant mutation bumps a version counter on the
  component it touches (the store, the room registry, the compressor bank,
  the fan sets). A snapshot is pinned to the version vector observed before
  collection and only accepted when the vector is unchanged afterwards, so
  every segment in the document describes the same moment.

* **Bounded generation.** Collection retries optimistically a fixed number
  of times. If the plant keeps mutating under it, the snapshotter takes the
  mutation gate's freeze side with a bounded wait and collects once more
  while writers are held back. If the freeze cannot be taken within the
  budget the request is refused, never served late or inconsistent.

* **Partial failure.** A component that fails to read is marked unavailable
  in its own segment; the rest of the document is still served. Only when
  every segment fails is the whole snapshot refused.

* **Cached by version, not generated on demand.** A snapshot is built once
  per plant version and then served from cache: the console endpoint and the
  CSV export read the same document for the same version, repeated reads
  cost O(1), and each generation has a stable identity the audit trail can
  reference. A partial snapshot is cached for its version like any other —
  within one version the plant has not moved, so every reader must see the
  same document, unavailable segments included.

* **Audited per generation.** Each generation (not each cache hit) appends a
  ``snapshot`` audit entry carrying the version, the segment availability
  and a description. The audit append itself bumps the store version by one;
  the cache key accounts for exactly that self-inflicted bump. The snapshot's
  own entry is therefore absent from its audit segment — a snapshot cannot
  contain the record of its own creation.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from coldstore.ns import control_line
from coldstore.store import MutationGate, store_state

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from .runtime import Runtime

# The mutation counters every consistent snapshot is validated against, in
# vector order. Everything else persists through the store and is covered by
# its counter.
VERSION_PARTS = ("store", "rooms", "compressors", "cold_fans", "evap_fans")

OK = "ok"
UNAVAILABLE = "unavailable"


class SnapshotUnavailable(Exception):
    """Raised when no consistent snapshot can be produced within budget."""


def plant_version(rt: "Runtime") -> tuple[int, ...]:
    """The version vector every consistent snapshot is validated against."""

    return (
        rt.store.version,
        rt.rooms.version,
        rt.compressors.version,
        rt.cold_fans.version,
        rt.evap_fans.version,
    )


def version_tag(version: tuple[int, ...]) -> str:
    """Compact dotted form of a version vector."""

    return ".".join(str(part) for part in version)


def _after_own_audit(version: tuple[int, ...]) -> tuple[int, ...]:
    """The vector once our own audit append has bumped the store counter."""

    return (version[0] + 1, *version[1:])


@dataclass(frozen=True)
class Snapshot:
    """One consistent, partially available view of the plant."""

    snapshot_id: str
    version: tuple[int, ...]
    generated_at: str
    duration_ms: float
    attempts: int
    frozen: bool
    complete: bool
    audited: bool
    availability: dict[str, str]
    errors: dict[str, str]
    segments: dict[str, object]
    description: str

    @property
    def tag(self) -> str:
        return version_tag(self.version)

    def as_dict(self) -> dict[str, object]:
        return {
            "meta": {
                "snapshot_id": self.snapshot_id,
                "version": self.tag,
                "versions": dict(zip(VERSION_PARTS, self.version)),
                "generated_at": self.generated_at,
                "duration_ms": round(self.duration_ms, 3),
                "attempts": self.attempts,
                "frozen": self.frozen,
                "complete": self.complete,
                "audited": self.audited,
                "availability": dict(self.availability),
                "errors": dict(self.errors),
                "description": self.description,
            },
            "segments": dict(self.segments),
        }

    def audit_detail(self) -> str:
        """The version, availability and description synced into the audit."""

        return json.dumps(
            {
                "snapshot_id": self.snapshot_id,
                "version": self.tag,
                "versions": dict(zip(VERSION_PARTS, self.version)),
                "availability": self.availability,
                "description": self.description,
                "duration_ms": round(self.duration_ms, 3),
                "attempts": self.attempts,
                "frozen": self.frozen,
                "complete": self.complete,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


Reader = Callable[["Runtime"], object]


def _read_pipeline(rt: "Runtime") -> object:
    return [step.as_dict() for step in control_line().steps()]


def _read_store(rt: "Runtime") -> object:
    return store_state(rt.store).as_dict()


def _read_rooms(rt: "Runtime") -> object:
    return rt.rooms.state().as_dict()


def _read_compressors(rt: "Runtime") -> object:
    return rt.compressors.state().as_dict()


def _read_supply(rt: "Runtime") -> object:
    return rt.supply.state().as_dict()


def _read_temp(rt: "Runtime") -> object:
    return rt.temp.state().as_dict()


def _read_ammonia(rt: "Runtime") -> object:
    return rt.ammonia.state().as_dict()


def _read_interlock(rt: "Runtime") -> object:
    return rt.interlock.state().as_dict()


def _read_cold_fans(rt: "Runtime") -> object:
    return rt.cold_fans.state().as_dict()


def _read_evap(rt: "Runtime") -> object:
    return rt.evap.state().as_dict()


def _read_defrost(rt: "Runtime") -> object:
    return rt.defrost.state().as_dict()


def _read_load(rt: "Runtime") -> object:
    return rt.load.state(rt.rooms).as_dict()


def _read_quota(rt: "Runtime") -> object:
    return rt.accumulator.state().as_dict()


def _read_audit(rt: "Runtime") -> object:
    return rt.auditor.state().as_dict()


def _read_schedule(rt: "Runtime") -> object:
    return rt.scheduler.state(rt.rooms).as_dict()


def segment_readers() -> list[tuple[str, Reader]]:
    """The snapshot segments in document order, rebuilt so tests can patch."""

    return [
        ("pipeline", _read_pipeline),
        ("store", _read_store),
        ("rooms", _read_rooms),
        ("compressors", _read_compressors),
        ("supply", _read_supply),
        ("temp", _read_temp),
        ("ammonia", _read_ammonia),
        ("interlock", _read_interlock),
        ("cold_fans", _read_cold_fans),
        ("evap", _read_evap),
        ("defrost", _read_defrost),
        ("load", _read_load),
        ("quota", _read_quota),
        ("audit", _read_audit),
        ("schedule", _read_schedule),
    ]


def collect_segments(
    rt: "Runtime",
) -> tuple[dict[str, object], dict[str, str], dict[str, str]]:
    """Read every segment, marking the ones that fail instead of raising."""

    segments: dict[str, object] = {}
    availability: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name, reader in segment_readers():
        try:
            segments[name] = reader(rt)
            availability[name] = OK
        except Exception as exc:  # noqa: BLE001 - one bad segment must not sink the rest
            segments[name] = None
            availability[name] = UNAVAILABLE
            errors[name] = str(exc) or type(exc).__name__
    return segments, availability, errors


def _require_some_segment(availability: dict[str, str]) -> None:
    """Refuse the whole snapshot when not a single segment could be read."""

    if not any(state == OK for state in availability.values()):
        raise SnapshotUnavailable("every segment failed to read")


class Snapshotter:
    """Generates, validates, caches and audits consistent snapshots.

    Generation is single-flight: concurrent callers for a stale version wait
    on one build and then share the cached document.
    """

    def __init__(
        self,
        rt: "Runtime",
        gate: MutationGate,
        *,
        max_attempts: int = 3,
        freeze_budget: float = 0.25,
    ) -> None:
        self._rt = rt
        self._gate = gate
        self._max_attempts = max_attempts
        self._freeze_budget = freeze_budget
        self._lock = threading.Lock()
        self._cached: tuple[tuple[int, ...], Snapshot] | None = None

    def current(self) -> Snapshot:
        """The snapshot for the current plant version, generating if stale."""

        cached = self._cached
        if cached is not None and cached[0] == plant_version(self._rt):
            return cached[1]
        with self._lock:
            cached = self._cached
            if cached is not None and cached[0] == plant_version(self._rt):
                return cached[1]
            return self._generate()

    def _generate(self) -> Snapshot:
        started = time.monotonic()
        attempts = 0
        for _ in range(self._max_attempts):
            attempts += 1
            before = plant_version(self._rt)
            segments, availability, errors = collect_segments(self._rt)
            _require_some_segment(availability)
            if plant_version(self._rt) == before:
                return self._finalize(
                    before, segments, availability, errors, attempts, False, started
                )
        return self._generate_frozen(attempts, started)

    def _generate_frozen(self, attempts: int, started: float) -> Snapshot:
        freeze = self._gate.freeze(self._freeze_budget)
        if freeze is None:
            raise SnapshotUnavailable(
                f"plant stayed busy beyond the {self._freeze_budget:.3f}s freeze budget"
            )
        with freeze:
            before = plant_version(self._rt)
            segments, availability, errors = collect_segments(self._rt)
            _require_some_segment(availability)
            return self._finalize(
                before, segments, availability, errors, attempts + 1, True, started
            )

    def _finalize(
        self,
        version: tuple[int, ...],
        segments: dict[str, object],
        availability: dict[str, str],
        errors: dict[str, str],
        attempts: int,
        frozen: bool,
        started: float,
    ) -> Snapshot:
        duration_ms = (time.monotonic() - started) * 1000.0
        readable = sum(1 for state in availability.values() if state == OK)
        snapshot = Snapshot(
            snapshot_id=uuid.uuid4().hex,
            version=version,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_ms=duration_ms,
            attempts=attempts,
            frozen=frozen,
            complete=readable == len(availability),
            audited=False,
            availability=availability,
            errors=errors,
            segments=segments,
            description=(
                f"snapshot v{version_tag(version)} segments={readable}/{len(availability)} ok "
                f"attempts={attempts} frozen={frozen} duration_ms={duration_ms:.1f}"
            ),
        )
        if not self._record_audit(snapshot):
            return snapshot
        snapshot = dataclasses.replace(snapshot, audited=True)
        served = plant_version(self._rt)
        if served == _after_own_audit(version):
            # Only our own audit append moved the version, so the document is
            # still the authoritative view for the version it now caches.
            self._cached = (served, snapshot)
        return snapshot

    def _record_audit(self, snapshot: Snapshot) -> bool:
        try:
            self._rt.auditor.record("snapshot", snapshot.audit_detail())
        except Exception:  # noqa: BLE001 - a broken audit trail must not sink the snapshot
            return False
        return True
