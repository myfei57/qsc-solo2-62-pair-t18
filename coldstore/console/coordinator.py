"""Versioned consistent snapshots with a hard generation budget.

Snapshots are cached per state version instead of regenerated on demand:
every version is captured at most once, and the console endpoint and the
CSV export serve the very same cached document.  While a snapshot is
collected the coordinator freezes component state by holding the
exclusive side of a gate; mutations pass the shared side, so a mutation
never interleaves with a collection and a collection always lines up
with one state version.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass


class SnapshotUnavailable(Exception):
    """The snapshot as a whole cannot be served; reject the request."""


class SnapshotVersionUnknown(Exception):
    """The requested version was never captured or has been evicted."""


@dataclass(frozen=True)
class SegmentAvailability:
    """Whether one component segment was readable for a snapshot."""

    available: bool
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"available": self.available, "error": self.error}


@dataclass(frozen=True)
class SnapshotDocument:
    """One immutable, versioned view of every component."""

    version: int
    generated_at: str
    partial: bool
    segments: Mapping[str, object]
    availability: Mapping[str, SegmentAvailability]
    describe: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "partial": self.partial,
            "availability": {
                name: state.as_dict() for name, state in self.availability.items()
            },
            "segments": dict(self.segments),
            "describe": dict(self.describe),
        }

    def unavailable_segments(self) -> list[str]:
        return [name for name, state in self.availability.items() if not state.available]


class _Freeze:
    """Shared/exclusive gate: mutations pass shared, snapshots exclusive.

    Snapshots win against waiting mutations so a frozen view always
    becomes available within the budget; a snapshot in turn never waits
    longer than the budget because in-flight mutations are short.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._exclusive = False
        self._shared = 0
        self._waiting_exclusive = 0

    @contextmanager
    def exclusive(self, timeout: float):
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._cond:
            self._waiting_exclusive += 1
            try:
                while self._exclusive or self._shared > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SnapshotUnavailable("state stayed busy past the snapshot budget")
                    self._cond.wait(remaining)
                self._exclusive = True
            finally:
                self._waiting_exclusive -= 1
        try:
            yield
        finally:
            with self._cond:
                self._exclusive = False
                self._cond.notify_all()

    @contextmanager
    def shared(self):
        with self._cond:
            while self._exclusive or self._waiting_exclusive > 0:
                self._cond.wait()
            self._shared += 1
        try:
            yield
        finally:
            with self._cond:
                self._shared -= 1
                if self._shared == 0:
                    self._cond.notify_all()


class _Flight:
    """One in-flight generation shared by concurrent waiters."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.document: SnapshotDocument | None = None
        self.error: Exception | None = None


class SnapshotCoordinator:
    """Generates, caches and audits consistent plant snapshots."""

    def __init__(
        self,
        readers: Sequence[tuple[str, Callable[[], object]]],
        describe: Callable[[], Mapping[str, str]] | None = None,
        record_audit: Callable[[str], None] | None = None,
        budget_seconds: float = 1.0,
        cache_versions: int = 8,
    ) -> None:
        if budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        if cache_versions <= 0:
            raise ValueError("cache_versions must be positive")
        self._readers = list(readers)
        self._describe = describe
        self._record_audit = record_audit
        self._budget = budget_seconds
        self._cache_versions = cache_versions
        self._freeze = _Freeze()
        self._lock = threading.Lock()
        self._version = 0
        self._cache: dict[int, SnapshotDocument] = {}
        self._flight: _Flight | None = None

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @contextmanager
    def mutation(self):
        """Hold the shared gate while a mutation runs; snapshots wait."""

        with self._freeze.shared():
            yield

    def note_mutation(self) -> int:
        """Advance the state version after a mutation committed."""

        with self._lock:
            self._version += 1
            return self._version

    def document(self, version: int | None = None) -> SnapshotDocument:
        """Serve the snapshot for ``version`` (the latest when omitted).

        Cache hits return immediately.  A miss starts one generation and
        concurrent callers for the same version ride the same flight, so
        each version is generated at most once no matter the load.
        """

        if version is not None and version < 0:
            raise SnapshotVersionUnknown(f"snapshot version {version} is unknown")
        with self._lock:
            current = self._version
            if version is not None and version > current:
                raise SnapshotVersionUnknown(f"snapshot version {version} is unknown")
            cached = self._cache.get(current if version is None else version)
            if cached is not None:
                return cached
            if version is not None and version != current:
                raise SnapshotVersionUnknown(f"snapshot version {version} has been evicted")
            flight = self._flight
            if flight is None:
                flight = _Flight()
                self._flight = flight
                owner = True
            else:
                owner = False
        if owner:
            try:
                document = self._generate()
            except Exception as exc:
                flight.error = exc
                with self._lock:
                    if self._flight is flight:
                        self._flight = None
                flight.done.set()
                raise
            with self._lock:
                self._cache[document.version] = document
                self._evict_locked()
                if self._flight is flight:
                    self._flight = None
            flight.document = document
            flight.done.set()
            return self._requested(document, version)
        if not flight.done.wait(self._budget):
            raise SnapshotUnavailable("snapshot generation missed its budget")
        if flight.error is not None:
            raise flight.error
        if flight.document is None:  # pragma: no cover - done implies a result
            raise SnapshotUnavailable("snapshot generation failed")
        return self._requested(flight.document, version)

    @staticmethod
    def _requested(document: SnapshotDocument, version: int | None) -> SnapshotDocument:
        """Reject a generated document when it missed an explicit version."""

        if version is not None and document.version != version:
            raise SnapshotVersionUnknown(f"snapshot version {version} has been evicted")
        return document

    def _generate(self) -> SnapshotDocument:
        """Collect every segment against one frozen state version."""

        deadline = time.monotonic() + self._budget
        with self._freeze.exclusive(self._budget):
            with self._lock:
                version = self._version
                cached = self._cache.get(version)
            if cached is not None:
                return cached
            generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            segments: dict[str, object] = {}
            availability: dict[str, SegmentAvailability] = {}
            for name, read in self._readers:
                if time.monotonic() > deadline:
                    segments[name] = None
                    availability[name] = SegmentAvailability(False, "deadline exceeded")
                    continue
                try:
                    segments[name] = read()
                except Exception as exc:  # noqa: BLE001 - one bad segment must not fail the rest
                    segments[name] = None
                    availability[name] = SegmentAvailability(
                        False, str(exc) or exc.__class__.__name__
                    )
                else:
                    availability[name] = SegmentAvailability(True)
            describe = self._describe_safely(deadline)
        document = SnapshotDocument(
            version=version,
            generated_at=generated_at,
            partial=any(not state.available for state in availability.values()),
            segments=segments,
            availability=availability,
            describe=describe,
        )
        self._audit(document)
        return document

    def _describe_safely(self, deadline: float) -> Mapping[str, str]:
        """Best effort describe lines; they never fail a snapshot."""

        if self._describe is None or time.monotonic() > deadline:
            return {}
        try:
            return dict(self._describe())
        except Exception:  # noqa: BLE001 - descriptions are advisory
            return {}

    def _audit(self, document: SnapshotDocument) -> None:
        """Sync version, segment availability and describe into the audit.

        Runs outside the freeze so the audit append cannot deadlock with
        mutations, and propagates failures: a snapshot that cannot be
        audited is rejected as a whole rather than served untracked.
        """

        if self._record_audit is None:
            return
        detail = json.dumps(
            {
                "version": document.version,
                "partial": document.partial,
                "unavailable": document.unavailable_segments(),
                "describe": dict(document.describe),
            },
            ensure_ascii=False,
        )
        self._record_audit(detail)

    def _evict_locked(self) -> None:
        """Bound the cache to the most recent versions."""

        while len(self._cache) > self._cache_versions:
            del self._cache[min(self._cache)]
