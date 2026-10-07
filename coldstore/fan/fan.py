"""Cold air fans grouped into a duty pair.

Every fan group carries a primary unit and at least one standby. Duty is
explicit rather than positional so a failed unit can be repaired and left
parked on standby while another takes over.
"""

from __future__ import annotations

import contextlib
import threading

from coldstore.store.gate import MutationGate

from .report import FanState, FanView

PRIMARY = "primary"
STANDBY = "standby"


class FanUnit:
    """Mutable state of one cold air fan."""

    def __init__(self, fan_id: str, role: str = STANDBY) -> None:
        self.id = fan_id
        self.role = role
        self.running = False
        self.faulted = False

    def snapshot(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role,
            "running": self.running,
            "faulted": self.faulted,
        }


class FanSet:
    """Thread safe set of fans for one cooling group."""

    def __init__(self) -> None:
        self._fans: dict[str, FanUnit] = {}
        self._lock = threading.RLock()
        self._version = 0
        self.mutation_gate: MutationGate | None = None

    @property
    def version(self) -> int:
        """Monotonic counter bumped by every fan set mutation."""

        with self._lock:
            return self._version

    def _mutation(self) -> contextlib.AbstractContextManager[None]:
        """Hold the shared mutation gate for one change, when wired."""

        gate = self.mutation_gate
        if gate is None:
            return contextlib.nullcontext()
        return gate.mutation()

    def add_fan(self, fan_id: str, role: str = STANDBY) -> FanUnit:
        with self._mutation():
            with self._lock:
                unit = FanUnit(fan_id=fan_id, role=role)
                self._fans[fan_id] = unit
                self._version += 1
                return unit

    def fan(self, fan_id: str) -> FanUnit | None:
        with self._lock:
            return self._fans.get(fan_id)

    def fans(self) -> list[FanUnit]:
        with self._lock:
            return [self._fans[key] for key in sorted(self._fans)]

    def fan_ids(self) -> list[str]:
        return [unit.id for unit in self.fans()]

    def count(self) -> int:
        with self._lock:
            return len(self._fans)

    def start(self, fan_id: str) -> None:
        with self._mutation():
            with self._lock:
                unit = self._require(fan_id)
                if unit.faulted:
                    raise ValueError(f"fan {fan_id} is faulted")
                unit.running = True
                self._version += 1

    def stop(self, fan_id: str) -> None:
        with self._mutation():
            with self._lock:
                self._require(fan_id).running = False
                self._version += 1

    def fail(self, fan_id: str) -> None:
        with self._mutation():
            with self._lock:
                unit = self._require(fan_id)
                unit.faulted = True
                unit.running = False
                self._version += 1

    def repair(self, fan_id: str) -> None:
        with self._mutation():
            with self._lock:
                unit = self._require(fan_id)
                unit.faulted = False
                if unit.role == PRIMARY:
                    unit.role = STANDBY
                self._version += 1

    def switch_over(self, fan_id: str) -> str:
        """Make one fan primary, demoting every other unit to standby."""

        with self._mutation():
            with self._lock:
                unit = self._require(fan_id)
                if unit.faulted:
                    raise ValueError(f"fan {fan_id} is faulted")
                for candidate in self._fans.values():
                    candidate.role = STANDBY
                unit.role = PRIMARY
                self._version += 1
                return unit.id

    def primary(self) -> str:
        with self._lock:
            for fan_id, unit in self._fans.items():
                if unit.role == PRIMARY:
                    return fan_id
            return ""

    def active(self) -> str:
        """The running primary fan, used as the duty unit."""

        with self._lock:
            for fan_id, unit in self._fans.items():
                if unit.role == PRIMARY and unit.running:
                    return fan_id
            return ""

    def running_ids(self) -> list[str]:
        return [unit.id for unit in self.fans() if unit.running]

    def faulted_ids(self) -> list[str]:
        return [unit.id for unit in self.fans() if unit.faulted]

    def healthy_standby(self) -> str:
        with self._lock:
            for fan_id, unit in self._fans.items():
                if unit.role == STANDBY and not unit.faulted:
                    return fan_id
            return ""

    def stop_all(self) -> None:
        with self._mutation():
            with self._lock:
                for unit in self._fans.values():
                    unit.running = False
                self._version += 1

    def state(self) -> FanState:
        fans = self.fans()
        return FanState(
            fans=[
                FanView(
                    id=unit.id,
                    role=unit.role,
                    running=unit.running,
                    faulted=unit.faulted,
                )
                for unit in fans
            ],
            primary=self.primary(),
            active=self.active(),
            count=len(fans),
        )

    def describe(self) -> str:
        state = self.state()
        return f"fans={state.count} primary={state.primary or ''} active={state.active or ''}"

    def _require(self, fan_id: str) -> FanUnit:
        unit = self._fans.get(fan_id)
        if unit is None:
            raise ValueError(f"fan {fan_id} not found")
        return unit
