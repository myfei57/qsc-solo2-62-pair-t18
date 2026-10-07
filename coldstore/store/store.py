"""Thread safe JSON document behind the control service."""

from __future__ import annotations

import contextlib
import json
import os
import threading

from .gate import MutationGate


class Store:
    """A file backed mapping from string keys to string values.

    Writes are atomic: the document is written to a sibling temporary file and
    then moved over the target path so a crash cannot leave a partial document.

    Every mutation bumps :attr:`version`, which the console snapshotter uses
    to validate that a collected snapshot saw one consistent plant state.
    When the runtime wires a :class:`MutationGate` in, each mutation also
    holds the gate's mutation side so a snapshot freeze can quiesce writes.
    """

    def __init__(self, path: str, data: dict[str, str] | None = None) -> None:
        self._path = path
        self._data: dict[str, str] = dict(data) if data else {}
        self._lock = threading.RLock()
        self._version = 0
        self.mutation_gate: MutationGate | None = None

    @property
    def path(self) -> str:
        return self._path

    @classmethod
    def open(cls, path: str) -> "Store":
        """Load a store from disk, creating an empty one when absent."""

        absolute = os.path.abspath(path)
        parent = os.path.dirname(absolute)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(absolute, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return cls(absolute)
        if not raw.strip():
            return cls(absolute)
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("store document must be a JSON object")
        return cls(absolute, {str(key): str(value) for key, value in loaded.items()})

    @property
    def version(self) -> int:
        """Monotonic counter bumped by every mutation of the document."""

        with self._lock:
            return self._version

    def _mutation(self) -> contextlib.AbstractContextManager[None]:
        """Hold the shared mutation gate for one write, when wired."""

        gate = self.mutation_gate
        if gate is None:
            return contextlib.nullcontext()
        return gate.mutation()

    def get(self, key: str) -> tuple[str, bool]:
        with self._lock:
            if key in self._data:
                return self._data[key], True
            return "", False

    def put(self, key: str, value: str) -> None:
        with self._mutation():
            with self._lock:
                self._data[key] = value
                self._version += 1
                self._save_locked()

    def delete(self, key: str) -> None:
        with self._mutation():
            with self._lock:
                if key not in self._data:
                    return
                del self._data[key]
                self._version += 1
                self._save_locked()

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._data)

    def count(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._mutation():
            with self._lock:
                self._data = {}
                self._version += 1
                self._save_locked()

    def close(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        raw = json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = f"{self._path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(raw)
        os.replace(temporary, self._path)
