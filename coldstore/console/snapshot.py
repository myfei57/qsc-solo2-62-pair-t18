"""Segment readers and coordinator wiring for the consistent snapshot."""

from __future__ import annotations

from collections.abc import Callable

from coldstore.ns import control_line
from coldstore.store import store_state

from .coordinator import SnapshotCoordinator
from .describe import collect as collect_describe
from .runtime import Runtime

Reader = Callable[[Runtime], object]


def _pipeline(rt: Runtime) -> object:
    return [step.as_dict() for step in control_line().steps()]


def _store(rt: Runtime) -> object:
    return store_state(rt.store).as_dict()


# Snapshot segments in collection order; one failure marks only that segment.
SEGMENTS: tuple[tuple[str, Reader], ...] = (
    ("pipeline", _pipeline),
    ("store", _store),
    ("rooms", lambda rt: rt.rooms.state().as_dict()),
    ("compressors", lambda rt: rt.compressors.state().as_dict()),
    ("supply", lambda rt: rt.supply.state().as_dict()),
    ("temp", lambda rt: rt.temp.state().as_dict()),
    ("ammonia", lambda rt: rt.ammonia.state().as_dict()),
    ("interlock", lambda rt: rt.interlock.state().as_dict()),
    ("cold_fans", lambda rt: rt.cold_fans.state().as_dict()),
    ("evap", lambda rt: rt.evap.state().as_dict()),
    ("defrost", lambda rt: rt.defrost.state().as_dict()),
    ("load", lambda rt: rt.load.state(rt.rooms).as_dict()),
    ("quota", lambda rt: rt.accumulator.state().as_dict()),
    ("audit", lambda rt: rt.auditor.state().as_dict()),
    ("schedule", lambda rt: rt.scheduler.state(rt.rooms).as_dict()),
)


def _bind(reader: Reader, rt: Runtime) -> Callable[[], object]:
    def read() -> object:
        return reader(rt)

    return read


def build_coordinator(
    rt: Runtime, budget_seconds: float = 1.0, cache_versions: int = 8
) -> SnapshotCoordinator:
    """Wire a versioned snapshot coordinator over one console runtime."""

    return SnapshotCoordinator(
        readers=[(name, _bind(reader, rt)) for name, reader in SEGMENTS],
        describe=lambda: collect_describe(rt),
        record_audit=lambda detail: rt.auditor.record("snapshot", detail),
        budget_seconds=budget_seconds,
        cache_versions=cache_versions,
    )
