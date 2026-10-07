"""Tests for the versioned, freeze-consistent plant snapshot."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest

from coldstore.console.coordinator import (
    SnapshotCoordinator,
    SnapshotUnavailable,
    SnapshotVersionUnknown,
)
from coldstore.console.http import Request
from coldstore.console.seed import seed_defaults
from coldstore.console.server import Server
from coldstore.store import Store


class CoordinatorCase(unittest.TestCase):
    def test_mutation_waits_while_snapshot_freezes_state(self) -> None:
        state = {"temp": -20.0, "stage": 1}
        first_read = threading.Event()
        release = threading.Event()

        def read_temp() -> dict[str, float]:
            first_read.set()
            release.wait(2.0)
            return {"temp": state["temp"]}

        def read_stage() -> dict[str, int]:
            return {"stage": state["stage"]}

        coordinator = SnapshotCoordinator(
            readers=[("temp", read_temp), ("stage", read_stage)],
            budget_seconds=5.0,
        )
        generated: dict[str, object] = {}
        thread = threading.Thread(
            target=lambda: generated.setdefault("doc", coordinator.document())
        )
        thread.start()
        self.assertTrue(first_read.wait(2.0))

        mutated = threading.Event()

        def mutate() -> None:
            with coordinator.mutation():
                state["temp"] = -5.0
                state["stage"] = 3
                mutated.set()

        writer = threading.Thread(target=mutate)
        writer.start()
        time.sleep(0.1)
        self.assertFalse(mutated.is_set())  # frozen while the snapshot reads
        release.set()
        thread.join(2.0)
        writer.join(2.0)
        self.assertTrue(mutated.is_set())

        doc = generated["doc"]
        self.assertEqual(doc.segments["temp"], {"temp": -20.0})
        self.assertEqual(doc.segments["stage"], {"stage": 1})
        self.assertFalse(doc.partial)

    def test_segment_failure_marks_only_that_segment(self) -> None:
        details: list[str] = []

        def boom() -> object:
            raise RuntimeError("sensor offline")

        coordinator = SnapshotCoordinator(
            readers=[("ok", lambda: {"v": 1}), ("bad", boom)],
            record_audit=details.append,
        )
        doc = coordinator.document()
        self.assertTrue(doc.partial)
        self.assertEqual(doc.segments["ok"], {"v": 1})
        self.assertIsNone(doc.segments["bad"])
        self.assertTrue(doc.availability["ok"].available)
        self.assertFalse(doc.availability["bad"].available)
        self.assertEqual(doc.availability["bad"].error, "sensor offline")
        self.assertEqual(doc.unavailable_segments(), ["bad"])

        detail = json.loads(details[0])
        self.assertEqual(detail["version"], 0)
        self.assertTrue(detail["partial"])
        self.assertEqual(detail["unavailable"], ["bad"])

    def test_deadline_marks_remaining_segments_unavailable(self) -> None:
        def slow() -> dict[str, int]:
            time.sleep(0.2)
            return {"v": 1}

        coordinator = SnapshotCoordinator(
            readers=[("slow", slow), ("later", lambda: {"v": 2})],
            budget_seconds=0.05,
        )
        doc = coordinator.document()
        self.assertTrue(doc.partial)
        self.assertEqual(doc.segments["slow"], {"v": 1})
        self.assertIsNone(doc.segments["later"])
        self.assertEqual(doc.availability["later"].error, "deadline exceeded")

    def test_busy_state_rejects_the_whole_snapshot(self) -> None:
        coordinator = SnapshotCoordinator(
            readers=[("a", lambda: {})], budget_seconds=0.05
        )
        errors: list[str] = []

        def generate() -> None:
            try:
                coordinator.document()
            except SnapshotUnavailable as exc:
                errors.append(str(exc))

        with coordinator.mutation():  # hold the shared gate like a long write
            thread = threading.Thread(target=generate)
            thread.start()
            thread.join(2.0)
        self.assertEqual(len(errors), 1)
        self.assertIn("budget", errors[0])

    def test_versions_are_cached_and_evicted_in_a_bounded_window(self) -> None:
        audits: list[str] = []
        coordinator = SnapshotCoordinator(
            readers=[("a", lambda: {"v": 1})],
            record_audit=audits.append,
            cache_versions=2,
        )
        first = coordinator.document()
        self.assertIs(coordinator.document(), first)  # cached, not regenerated
        self.assertEqual(len(audits), 1)

        coordinator.note_mutation()
        second = coordinator.document()
        self.assertEqual(second.version, 1)
        self.assertIsNot(second, first)
        self.assertIs(coordinator.document(0), first)  # old version still cached

        coordinator.note_mutation()
        third = coordinator.document()
        self.assertEqual(third.version, 2)
        self.assertEqual(len(audits), 3)  # one audit entry per generation

        with self.assertRaises(SnapshotVersionUnknown):
            coordinator.document(0)  # evicted from the bounded window
        self.assertIs(coordinator.document(1), second)
        with self.assertRaises(SnapshotVersionUnknown):
            coordinator.document(3)  # future version
        with self.assertRaises(SnapshotVersionUnknown):
            coordinator.document(-1)

    def test_concurrent_readers_share_one_generation(self) -> None:
        calls = {"count": 0}
        lock = threading.Lock()

        def read() -> dict[str, int]:
            with lock:
                calls["count"] += 1
            time.sleep(0.05)
            return {"v": 1}

        coordinator = SnapshotCoordinator(readers=[("a", read)])
        results: list[object] = []
        threads = [
            threading.Thread(target=lambda: results.append(coordinator.document()))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2.0)
        self.assertEqual(len(results), 8)
        self.assertEqual(calls["count"], 1)
        self.assertTrue(all(doc is results[0] for doc in results))

    def test_audit_detail_carries_version_availability_and_describe(self) -> None:
        details: list[str] = []
        coordinator = SnapshotCoordinator(
            readers=[("a", lambda: {"v": 1})],
            describe=lambda: {"a": "a line"},
            record_audit=details.append,
        )
        doc = coordinator.document()
        self.assertEqual(doc.describe, {"a": "a line"})
        detail = json.loads(details[0])
        self.assertEqual(detail["version"], doc.version)
        self.assertFalse(detail["partial"])
        self.assertEqual(detail["unavailable"], [])
        self.assertEqual(detail["describe"], {"a": "a line"})


class SnapshotHttpCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store.open(f"{self._tmp.name}/state.json")
        seed_defaults(self.store)
        self.server = Server(self.store)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def call(self, method: str, path: str, payload: dict | None = None):
        response = self.server.dispatch(method, path, payload)
        body = response.body.decode("utf-8")
        parsed = json.loads(body) if response.content_type.startswith("application/json") else body
        return response.status, parsed

    def test_mutations_advance_the_snapshot_version(self) -> None:
        status, first = self.call("GET", "/snapshot")
        self.assertEqual(status, 200)
        self.assertEqual(first["version"], 0)

        self.call("POST", "/room/add", {"id": "r1", "zone": 1, "temp": -20.0})
        status, second = self.call("GET", "/snapshot")
        self.assertEqual(second["version"], 1)
        self.assertEqual(second["segments"]["rooms"]["count"], 1)

        status, again = self.call("GET", "/snapshot")  # cache hit, same version
        self.assertEqual(again["version"], 1)
        self.assertEqual(again["generated_at"], second["generated_at"])

        self.call("POST", "/quota/check", {"resource": "runtime", "used": 1.0, "limit": 5.0})
        status, third = self.call("GET", "/snapshot")  # read-only POST: no bump
        self.assertEqual(third["version"], 1)

    def test_console_and_export_serve_the_same_document(self) -> None:
        status, snap = self.call("GET", "/snapshot")
        self.assertEqual(status, 200)
        status, csv = self.call("GET", "/export/snapshot")
        self.assertEqual(status, 200)
        self.assertIn("section,key,value", csv)
        self.assertIn(f"meta,version,{snap['version']}", csv)
        self.assertIn(f"meta,generated_at,{snap['generated_at']}", csv)
        self.assertIn("meta,partial,False", csv)
        self.assertIn("availability,temp,available", csv)
        self.assertIn("describe,pipeline,", csv)
        self.assertIn("segment,segments.store.size,", csv)

    def test_explicit_version_reads_and_unknown_versions(self) -> None:
        status, snap = self.call("GET", "/snapshot")
        version = snap["version"]
        self.call("POST", "/quota/add", {"amount": 5.0})

        status, older = self.call("GET", "/snapshot", {"version": version})
        self.assertEqual(status, 200)
        self.assertEqual(older["version"], version)
        self.assertEqual(older["generated_at"], snap["generated_at"])

        status, csv = self.call("GET", "/export/snapshot", {"version": version})
        self.assertIn(f"meta,version,{version}", csv)

        status, error = self.call("GET", "/snapshot", {"version": 999})
        self.assertEqual(status, 404)
        self.assertIn("999", error["error"])
        status, error = self.call("GET", "/snapshot", {"version": "abc"})
        self.assertEqual(status, 400)

        request = Request(
            method="GET",
            path="/snapshot",
            query={"version": [str(version)]},
            payload={},
        )
        response = self.server.respond(request)
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body)["version"], version)

    def test_segment_failure_is_partial_not_fatal(self) -> None:
        class BrokenCollector:
            def state(self) -> object:
                raise RuntimeError("modbus timeout")

            def describe(self) -> str:
                return "temp unreadable"

        self.server.runtime.temp = BrokenCollector()
        status, snap = self.call("GET", "/snapshot")
        self.assertEqual(status, 200)
        self.assertTrue(snap["partial"])
        self.assertFalse(snap["availability"]["temp"]["available"])
        self.assertIn("modbus timeout", snap["availability"]["temp"]["error"])
        self.assertIsNone(snap["segments"]["temp"])
        self.assertTrue(snap["availability"]["rooms"]["available"])
        self.assertEqual(snap["describe"]["temp"], "temp unreadable")

        status, csv = self.call("GET", "/export/snapshot")
        self.assertIn("availability,temp,unavailable: modbus timeout", csv)

    def test_generation_is_audited_once_per_version(self) -> None:
        status, snap = self.call("GET", "/snapshot")
        self.assertEqual(status, 200)

        status, body = self.call("POST", "/audit/filter", {"kind": "snapshot"})
        self.assertEqual(len(body["entries"]), 1)
        detail = json.loads(body["entries"][0]["detail"])
        self.assertEqual(detail["version"], snap["version"])
        self.assertFalse(detail["partial"])
        self.assertEqual(detail["unavailable"], [])
        self.assertIn("pipeline", detail["describe"])

        self.call("GET", "/snapshot")  # cache hit: no extra audit entry
        status, body = self.call("POST", "/audit/filter", {"kind": "snapshot"})
        self.assertEqual(len(body["entries"]), 1)

        self.call("POST", "/quota/add", {"amount": 1.0})
        status, snap = self.call("GET", "/snapshot")
        status, body = self.call("POST", "/audit/filter", {"kind": "snapshot"})
        self.assertEqual(len(body["entries"]), 2)
        detail = json.loads(body["entries"][1]["detail"])
        self.assertEqual(detail["version"], snap["version"])

    def test_whole_snapshot_failures_are_rejected(self) -> None:
        class BusySnapshots:
            def document(self, version=None):
                raise SnapshotUnavailable("state stayed busy past the snapshot budget")

        self.server.snapshots = BusySnapshots()
        status, body = self.call("GET", "/snapshot")
        self.assertEqual(status, 503)
        self.assertIn("budget", body["error"])
        status, body = self.call("GET", "/export/snapshot")
        self.assertEqual(status, 503)

        class GoneSnapshots:
            def document(self, version=None):
                raise SnapshotVersionUnknown("snapshot version 2 has been evicted")

        self.server.snapshots = GoneSnapshots()
        status, body = self.call("GET", "/snapshot")
        self.assertEqual(status, 404)
        self.assertIn("evicted", body["error"])


if __name__ == "__main__":
    unittest.main()
