"""Unit tests for the anti-collision zone state machine."""

from __future__ import annotations

import unittest

from cranefleet.geometry import Point
from cranefleet.store import MemoryStore
from cranefleet.zone import (
    BASIS_OCCUPANCY_CLEARED,
    BASIS_POSITION_KNOWN,
    BASIS_SAFE_SEPARATION,
    Conflict,
    ZoneState,
    boundary_area,
    coverage_check,
    crane_latched,
    detect_conflicts,
    evaluate_releases,
    is_latched,
    latch_generation,
    latch_report,
    latched_cranes,
    latest_registration_seq,
    occupancy_version,
    occupy,
    occupants_snapshot,
    persist,
    reason_peer_unlocated,
    reason_too_close,
    register_conflict,
    registration_count,
    registration_seq_of,
    release_ledger_of,
    restore,
    separation,
    vacate,
)

BOUNDARY = [Point(0.0, 0.0), Point(100.0, 0.0), Point(100.0, 100.0), Point(0.0, 100.0)]
SAFE = 12.0


class RecordingNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def on_latch_released(self, zone_id, crane_id, generation) -> None:
        self.calls.append((zone_id, crane_id, generation))


class ZoneOccupancyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = ZoneState("zone-1", "吊装作业区", BOUNDARY)

    def test_occupy_and_vacate(self) -> None:
        self.assertEqual(occupy(self.zone, ""), 0)
        self.assertEqual(occupy(self.zone, "crane-1"), 1)
        self.assertEqual(occupy(self.zone, "crane-2"), 2)
        self.assertEqual(occupy(self.zone, "crane-1"), 3)
        self.assertTrue(self.zone.occupied_by("crane-1"))
        self.assertEqual(self.zone.occupant_count(), 2)
        self.assertEqual(registration_count(self.zone), 3)
        self.assertEqual(latest_registration_seq(self.zone), 3)
        self.assertEqual(registration_seq_of(self.zone, "crane-1"), 3)
        self.assertEqual(occupancy_version(self.zone), 3)
        vacate(self.zone, "crane-1")
        self.assertEqual(occupants_snapshot(self.zone), ["crane-2"])
        self.assertEqual(occupancy_version(self.zone), 4)
        vacate(self.zone, "crane-1")
        self.assertEqual(occupancy_version(self.zone), 4)

    def test_persist_and_restore(self) -> None:
        store = MemoryStore()
        occupy(self.zone, "crane-1")
        self.assertFalse(self.zone.is_durable())
        persist(self.zone, store)
        self.assertTrue(self.zone.is_durable())
        self.assertEqual(self.zone.durable_revision(), 1)
        occupy(self.zone, "crane-2")
        persist(self.zone, store)
        reloaded = ZoneState("zone-1", "stale", [])
        restore(reloaded, store)
        self.assertEqual(reloaded.name, "吊装作业区")
        self.assertTrue(reloaded.is_durable())
        self.assertEqual(sorted(occupants_snapshot(reloaded)), ["crane-1", "crane-2"])
        self.assertEqual(reloaded.boundary_copy(), BOUNDARY)

    def test_restore_without_snapshot_is_not_durable(self) -> None:
        zone = ZoneState("zone-9", "empty", BOUNDARY)
        restore(zone, MemoryStore())
        self.assertFalse(zone.is_durable())

    def test_conflict_detection_and_registration(self) -> None:
        occupy(self.zone, "crane-1")
        occupy(self.zone, "crane-2")
        positions = {"crane-1": Point(0.0, 0.0), "crane-2": Point(5.0, 0.0)}
        conflicts = detect_conflicts(self.zone, positions, 12.0)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].crane_ids, ("crane-1", "crane-2"))
        self.assertEqual(conflicts[0].to_dict()["zone_id"], "zone-1")
        self.assertEqual(detect_conflicts(self.zone, {"crane-1": Point(0.0, 0.0)}, 12.0), [])

    def test_register_conflict_latches_each_crane(self) -> None:
        register_conflict(self.zone, ("crane-1", "crane-2", ""))
        self.assertTrue(self.zone.is_conflict())
        self.assertTrue(is_latched(self.zone))
        self.assertEqual(latched_cranes(self.zone), ["crane-1", "crane-2"])
        self.assertTrue(crane_latched(self.zone, "crane-1"))
        self.assertFalse(crane_latched(self.zone, "crane-9"))
        # A repeated report keeps the original latch order.
        register_conflict(self.zone, ("crane-2", "crane-1"))
        self.assertEqual(latched_cranes(self.zone), ["crane-1", "crane-2"])
        self.assertEqual(latch_generation(self.zone), 0)

    def test_clear_while_still_occupying_holds_every_crane(self) -> None:
        occupy(self.zone, "crane-1")
        occupy(self.zone, "crane-2")
        register_conflict(self.zone, ("crane-1", "crane-2"))
        positions = {"crane-1": Point(0.0, 0.0), "crane-2": Point(5.0, 0.0)}
        notifier = RecordingNotifier()
        decision = evaluate_releases(self.zone, positions, SAFE, notifier)
        self.assertEqual(decision.released, ())
        self.assertEqual(notifier.calls, [])
        self.assertTrue(is_latched(self.zone))
        self.assertEqual(latch_generation(self.zone), 0)
        withheld = {entry.crane_id: entry for entry in decision.withheld}
        self.assertEqual(set(withheld), {"crane-1", "crane-2"})
        crane_1 = withheld["crane-1"]
        self.assertIn("still-occupying", crane_1.reasons)
        self.assertIn(reason_too_close("crane-2"), crane_1.reasons)
        self.assertEqual(crane_1.blockers, ("crane-2",))
        self.assertAlmostEqual(crane_1.min_distance, 5.0)
        self.assertAlmostEqual(crane_1.safe_distance, SAFE)
        self.assertTrue(crane_1.occupying)

    def test_release_one_crane_while_peer_still_latched(self) -> None:
        occupy(self.zone, "crane-1")
        occupy(self.zone, "crane-2")
        register_conflict(self.zone, ("crane-1", "crane-2"))
        positions = {"crane-1": Point(0.0, 0.0), "crane-2": Point(80.0, 0.0)}
        vacate(self.zone, "crane-1")
        notifier = RecordingNotifier()
        decision = evaluate_releases(self.zone, positions, SAFE, notifier)
        self.assertEqual(decision.released_ids(), ["crane-1"])
        self.assertTrue(is_latched(self.zone))
        self.assertFalse(self.zone.is_conflict())
        self.assertEqual(latched_cranes(self.zone), ["crane-2"])
        self.assertFalse(crane_latched(self.zone, "crane-1"))
        entry = decision.released[0]
        self.assertEqual(entry.order, 1)
        self.assertEqual(entry.generation, 1)
        self.assertEqual(entry.basis, (BASIS_POSITION_KNOWN, BASIS_OCCUPANCY_CLEARED, BASIS_SAFE_SEPARATION))
        self.assertAlmostEqual(entry.min_distance, 80.0)
        self.assertEqual(notifier.calls, [("zone-1", "crane-1", 1)])
        self.assertEqual(decision.withheld[0].crane_id, "crane-2")
        self.assertEqual(decision.withheld[0].reasons, ("still-occupying",))
        ledger = release_ledger_of(self.zone)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0].to_dict()["crane_id"], "crane-1")
        self.assertEqual(ledger[0].to_dict()["basis"], list(entry.basis))

    def test_second_pass_releases_remaining_crane(self) -> None:
        occupy(self.zone, "crane-1")
        occupy(self.zone, "crane-2")
        register_conflict(self.zone, ("crane-1", "crane-2"))
        positions = {"crane-1": Point(0.0, 0.0), "crane-2": Point(80.0, 0.0)}
        vacate(self.zone, "crane-1")
        first = evaluate_releases(self.zone, positions, SAFE)
        self.assertEqual(first.generation, 1)
        self.assertTrue(is_latched(self.zone))
        vacate(self.zone, "crane-2")
        second = evaluate_releases(self.zone, positions, SAFE)
        self.assertEqual(second.released_ids(), ["crane-2"])
        self.assertEqual(second.generation, 2)
        self.assertFalse(is_latched(self.zone))
        self.assertFalse(self.zone.is_conflict())
        ledger = release_ledger_of(self.zone)
        self.assertEqual([entry.crane_id for entry in ledger], ["crane-1", "crane-2"])
        self.assertEqual([entry.order for entry in ledger], [1, 2])

    def test_release_follows_latch_order(self) -> None:
        occupy(self.zone, "crane-1")
        occupy(self.zone, "crane-2")
        register_conflict(self.zone, ("crane-2",))
        register_conflict(self.zone, ("crane-1",))
        vacate(self.zone, "crane-1")
        vacate(self.zone, "crane-2")
        positions = {"crane-1": Point(0.0, 0.0), "crane-2": Point(80.0, 0.0)}
        decision = evaluate_releases(self.zone, positions, SAFE)
        self.assertEqual(decision.released_ids(), ["crane-2", "crane-1"])
        self.assertEqual([entry.latch_seq for entry in decision.released], [1, 2])

    def test_missing_position_keeps_crane_latched(self) -> None:
        register_conflict(self.zone, ("crane-1", "crane-2"))
        vacate(self.zone, "crane-1")
        vacate(self.zone, "crane-2")
        # No surveyed position at all.
        decision = evaluate_releases(self.zone, {}, SAFE)
        self.assertEqual(decision.released, ())
        withheld = {entry.crane_id: entry for entry in decision.withheld}
        self.assertIn("missing-position", withheld["crane-1"].reasons)
        self.assertIn(reason_peer_unlocated("crane-2"), withheld["crane-1"].reasons)
        # crane-2 alone still lacks its own position.
        decision = evaluate_releases(self.zone, {"crane-1": Point(0.0, 0.0)}, SAFE)
        withheld = {entry.crane_id: entry for entry in decision.withheld}
        self.assertEqual(decision.released, ())
        self.assertNotIn("missing-position", withheld["crane-1"].reasons)
        self.assertIn("missing-position", withheld["crane-2"].reasons)
        # With both fresh and far apart, both release in the same pass.
        positions = {"crane-1": Point(0.0, 0.0), "crane-2": Point(80.0, 0.0)}
        decision = evaluate_releases(self.zone, positions, SAFE)
        self.assertEqual(decision.released_ids(), ["crane-1", "crane-2"])
        self.assertEqual(decision.generation, 1)

    def test_close_pair_vacating_together_stays_latched(self) -> None:
        register_conflict(self.zone, ("crane-1", "crane-2"))
        vacate(self.zone, "crane-1")
        vacate(self.zone, "crane-2")
        positions = {"crane-1": Point(0.0, 0.0), "crane-2": Point(5.0, 0.0)}
        decision = evaluate_releases(self.zone, positions, SAFE)
        self.assertEqual(decision.released, ())
        withheld = {entry.crane_id: entry for entry in decision.withheld}
        self.assertIn(reason_too_close("crane-2"), withheld["crane-1"].reasons)
        self.assertIn(reason_too_close("crane-1"), withheld["crane-2"].reasons)
        self.assertTrue(is_latched(self.zone))

    def test_latch_report_is_live_and_non_mutating(self) -> None:
        occupy(self.zone, "crane-1")
        occupy(self.zone, "crane-2")
        register_conflict(self.zone, ("crane-1", "crane-2"))
        close = {"crane-1": Point(0.0, 0.0), "crane-2": Point(5.0, 0.0)}
        report = latch_report(self.zone, close, SAFE)
        self.assertEqual([entry.crane_id for entry in report], ["crane-1", "crane-2"])
        self.assertTrue(all(entry.reasons for entry in report))
        self.assertTrue(is_latched(self.zone))
        # A crane that is now safe shows empty reasons but is still latched
        # until the next evaluation pass actually releases it.
        vacate(self.zone, "crane-2")
        far = {"crane-1": Point(0.0, 0.0), "crane-2": Point(80.0, 0.0)}
        report = latch_report(self.zone, far, SAFE)
        by_id = {entry.crane_id: entry for entry in report}
        self.assertEqual(by_id["crane-2"].reasons, ())
        self.assertTrue(crane_latched(self.zone, "crane-2"))

    def test_latch_state_persists_across_restore(self) -> None:
        store = MemoryStore()
        register_conflict(self.zone, ("crane-1", "crane-2"))
        positions = {"crane-1": Point(0.0, 0.0), "crane-2": Point(80.0, 0.0)}
        vacate(self.zone, "crane-1")
        evaluate_releases(self.zone, positions, SAFE)
        persist(self.zone, store)
        reloaded = ZoneState("zone-1", "stale", [])
        restore(reloaded, store)
        self.assertEqual(latched_cranes(reloaded), ["crane-2"])
        self.assertTrue(is_latched(reloaded))
        self.assertEqual(latch_generation(reloaded), 1)
        ledger = release_ledger_of(reloaded)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0].crane_id, "crane-1")
        self.assertAlmostEqual(ledger[0].min_distance, 80.0)
        vacate(reloaded, "crane-2")
        decision = evaluate_releases(reloaded, positions, SAFE)
        self.assertEqual(decision.released_ids(), ["crane-2"])
        self.assertEqual(decision.generation, 2)

    def test_legacy_snapshot_seeds_latches_from_occupants(self) -> None:
        store = MemoryStore()
        store.save(
            "zone/zone-1",
            (
                b'{"id": "zone-1", "name": "old", "boundary": '
                b'[{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 0.0}, '
                b'{"x": 100.0, "y": 100.0}, {"x": 0.0, "y": 100.0}], '
                b'"revision": 3, "occupants": ["crane-1", "crane-2"], '
                b'"registrations": [], "conflict": true, "latch": true}'
            ),
        )
        reloaded = ZoneState("zone-1", "stale", [])
        restore(reloaded, store)
        self.assertEqual(latched_cranes(reloaded), ["crane-1", "crane-2"])
        self.assertTrue(is_latched(reloaded))

    def test_coverage_and_area(self) -> None:
        positions = {"crane-1": Point(50.0, 50.0), "crane-2": Point(150.0, 50.0)}
        coverage = coverage_check(self.zone, positions)
        self.assertEqual(coverage.inside, ("crane-1",))
        self.assertEqual(coverage.outside, ("crane-2",))
        self.assertEqual(coverage.inside_bounds, ("crane-1",))
        self.assertAlmostEqual(coverage.area, 10000.0)
        self.assertGreater(coverage.expanded_area, coverage.area)
        self.assertTrue(coverage.expanded_overlap)
        self.assertIn("center", coverage.to_dict())
        self.assertAlmostEqual(boundary_area(self.zone), 10000.0)

    def test_separation_report(self) -> None:
        occupy(self.zone, "crane-1")
        occupy(self.zone, "crane-2")
        report = separation(
            self.zone, {"crane-1": Point(0.0, 0.0), "crane-2": Point(3.0, 4.0)}
        )
        self.assertAlmostEqual(report.distance, 5.0)
        self.assertTrue(report.fresh)
        self.assertIn("positions", report.to_dict())
        lone = separation(self.zone, {"crane-1": Point(0.0, 0.0)})
        self.assertEqual(lone.distance, -1.0)

    def test_conflict_dataclass_serialisation(self) -> None:
        conflict = Conflict(zone_id="zone-1", crane_ids=("a", "b"))
        self.assertEqual(conflict.to_dict(), {"zone_id": "zone-1", "crane_ids": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
