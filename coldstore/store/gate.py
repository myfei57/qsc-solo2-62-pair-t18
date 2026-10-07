"""Coordination between control mutations and consistent snapshot freezes."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class MutationGate:
    """A read-write gate that lets a snapshot freeze quiesce the plant.

    Every state change holds the *mutation* (read) side for its duration, so
    overlapping mutations never block each other. A snapshot that cannot be
    validated optimistically takes the *freeze* (write) side: it waits for
    in-flight mutations to drain and holds new ones back while one bounded
    collection runs.

    Two refinements keep the gate live under load:

    * Once a freeze is requested, new mutations wait instead of slipping in
      ahead of it, so a busy plant cannot starve the freezer.
    * Both sides are re-entrant for the thread already holding them: a
      mutation that fans out across components (defrost touches fans, rooms
      and the store) nests without deadlocking against a pending freeze, and
      the snapshotter appends its own audit entry while frozen.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers: dict[int, int] = {}
        self._writer: int | None = None
        self._freeze_waiting = 0

    @contextmanager
    def mutation(self) -> Iterator[None]:
        """Hold the mutation side for one state change."""

        ident = threading.get_ident()
        with self._cond:
            if self._writer == ident:
                counted = False
            else:
                if ident not in self._readers:
                    while self._writer is not None or self._freeze_waiting > 0:
                        self._cond.wait()
                    self._readers[ident] = 0
                self._readers[ident] += 1
                counted = True
        try:
            yield
        finally:
            if counted:
                with self._cond:
                    self._readers[ident] -= 1
                    if self._readers[ident] == 0:
                        del self._readers[ident]
                        if not self._readers:
                            self._cond.notify_all()

    def freeze(self, timeout: float) -> "Freeze | None":
        """Take the freeze side, giving up once ``timeout`` seconds elapse."""

        ident = threading.get_ident()
        deadline = time.monotonic() + timeout
        with self._cond:
            self._freeze_waiting += 1
            try:
                while self._writer is not None or self._readers:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
                self._writer = ident
            finally:
                self._freeze_waiting -= 1
                if self._freeze_waiting == 0:
                    self._cond.notify_all()
        return Freeze(self, ident)

    def _thaw(self, ident: int) -> None:
        with self._cond:
            if self._writer == ident:
                self._writer = None
                self._cond.notify_all()


class Freeze:
    """The held freeze side of a :class:`MutationGate`."""

    def __init__(self, gate: MutationGate, ident: int) -> None:
        self._gate = gate
        self._ident = ident
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            self._gate._thaw(self._ident)

    def __enter__(self) -> "Freeze":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
