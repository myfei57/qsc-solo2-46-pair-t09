"""Anti-collision zone state.

Every zone owns the occupancy registrations reported by the cranes working
inside it, the conflict flag raised when two cranes get too close, and the
latch that keeps a swing blocked until the conflict is cleared. The latch
releases crane by crane: each crane is assessed on its own conditions, the
release order and basis are recorded in the release ledger, and any crane
that falls short stays latched with the missing conditions noted. The zone
is the component whose state must be durable before any swing is allowed,
so the persistence helpers live here rather than in the console.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .geometry import Point, Polygon
from .errors import StoreNotFound
from .store import Store

HELD_CONFLICT = "冲突未解除"
BASIS_VACATED = "占用已退出分区"
BASIS_ALONE = "区内无其他占用车辆"
MISSING_OWN_POSITION = "缺少实时位置，无法核验间距"


@dataclass(frozen=True)
class ReleaseEntry:
    """One crane released from the latch, with its release order and basis."""

    crane_id: str
    generation: int
    seq: int
    basis: str

    def to_dict(self) -> dict[str, object]:
        return {
            "crane_id": self.crane_id,
            "generation": self.generation,
            "seq": self.seq,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class HeldCrane:
    """One crane still latched, with the conditions it has not met yet."""

    crane_id: str
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"crane_id": self.crane_id, "missing": list(self.missing)}


@dataclass(frozen=True)
class ReleaseOutcome:
    """The result of one clear attempt, in the order releases were granted."""

    generation: int
    releases: tuple[ReleaseEntry, ...]
    held: tuple[HeldCrane, ...]

    @property
    def released(self) -> tuple[str, ...]:
        return tuple(entry.crane_id for entry in self.releases)

    @property
    def latched(self) -> bool:
        return bool(self.held)


@dataclass(frozen=True)
class Conflict:
    """A pair of cranes whose clearance fell below the safe distance."""

    zone_id: str
    crane_ids: tuple[str, str]

    def to_dict(self) -> dict[str, object]:
        return {"zone_id": self.zone_id, "crane_ids": list(self.crane_ids)}


@dataclass(frozen=True)
class Coverage:
    """How the current crane positions sit inside the zone boundary."""

    inside: tuple[str, ...]
    outside: tuple[str, ...]
    inside_bounds: tuple[str, ...]
    area: float
    bounds_width: float
    bounds_height: float
    center: Point
    expanded_area: float
    expanded_overlap: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "inside": list(self.inside),
            "outside": list(self.outside),
            "inside_bounds": list(self.inside_bounds),
            "area": self.area,
            "bounds_width": self.bounds_width,
            "bounds_height": self.bounds_height,
            "center": self.center.to_dict(),
            "expanded_area": self.expanded_area,
            "expanded_overlap": self.expanded_overlap,
        }


@dataclass(frozen=True)
class DistanceReport:
    """The closest clearance between the cranes currently inside the zone."""

    distance: float
    positions: dict[str, Point]
    revision: int
    fresh: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "distance": self.distance,
            "positions": {key: point.to_dict() for key, point in self.positions.items()},
            "revision": self.revision,
            "fresh": self.fresh,
        }


class ConflictNotifier(Protocol):
    """Receives the release notification once a zone conflict clears."""

    def on_conflict_cleared(
        self,
        zone_id: str,
        crane_ids: Sequence[str],
        generation: int,
        conflicts: Sequence[str],
    ) -> None:
        ...


class ZoneState:
    """Mutable state of a single anti-collision zone."""

    def __init__(self, zone_id: str, name: str, boundary: Sequence[Point]) -> None:
        self._lock = threading.Lock()
        self.id = zone_id
        self.name = name
        self.boundary: list[Point] = list(boundary)
        self.revision = 0
        self.durable = False
        self.occupants: dict[str, bool] = {}
        self.registrations: list[str] = []
        self.conflict = False
        self.latch = False
        self.latch_generation = 0
        self.conflict_pairs: list[str] = []
        self.cached_positions: dict[str, Point] = {}
        self.registration_seq = 0
        self.registration_seq_by_crane: dict[str, int] = {}
        self.release_ledger: list[ReleaseEntry] = []
        self.held_cranes: dict[str, tuple[str, ...]] = {}
        self.release_seq = 0
        self.occupancy_version = 0

    def occupied_by(self, crane_id: str) -> bool:
        with self._lock:
            return self.occupants.get(crane_id, False)

    def occupant_count(self) -> int:
        with self._lock:
            return len(self.occupants)

    def boundary_copy(self) -> list[Point]:
        with self._lock:
            return list(self.boundary)

    def durable_revision(self) -> int:
        with self._lock:
            return self.revision

    def is_durable(self) -> bool:
        with self._lock:
            return self.durable

    def is_conflict(self) -> bool:
        with self._lock:
            return self.conflict


def _zone_key(zone_id: str) -> str:
    return "zone/" + zone_id


def occupy(zone: ZoneState, crane_id: str) -> int:
    """Register a crane inside the zone and return its registration sequence."""

    if not crane_id:
        return 0
    with zone._lock:  # noqa: SLF001 - module-level functions share zone state
        zone.occupants.setdefault(crane_id, True)
        zone.registrations.append(crane_id)
        zone.registration_seq += 1
        zone.registration_seq_by_crane[crane_id] = zone.registration_seq
        zone.occupancy_version += 1
        return zone.registration_seq


def vacate(zone: ZoneState, crane_id: str) -> None:
    """Remove a crane from the zone once it leaves."""

    with zone._lock:  # noqa: SLF001
        if zone.occupants.pop(crane_id, None) is not None:
            zone.occupancy_version += 1


def occupants_snapshot(zone: ZoneState) -> list[str]:
    with zone._lock:  # noqa: SLF001
        return list(zone.occupants)


def registration_count(zone: ZoneState) -> int:
    with zone._lock:  # noqa: SLF001
        return len(zone.registrations)


def registration_seq_of(zone: ZoneState, crane_id: str) -> int:
    with zone._lock:  # noqa: SLF001
        return zone.registration_seq_by_crane.get(crane_id, 0)


def latest_registration_seq(zone: ZoneState) -> int:
    with zone._lock:  # noqa: SLF001
        return zone.registration_seq


def is_latched(zone: ZoneState) -> bool:
    with zone._lock:  # noqa: SLF001
        return zone.latch


def latch_generation(zone: ZoneState) -> int:
    with zone._lock:  # noqa: SLF001
        return zone.latch_generation


def pending_releases(zone: ZoneState) -> list[ReleaseEntry]:
    with zone._lock:  # noqa: SLF001
        return list(zone.release_ledger)


def held_snapshot(zone: ZoneState) -> list[HeldCrane]:
    """Return the cranes still latched, each with its missing conditions."""

    with zone._lock:  # noqa: SLF001
        return [
            HeldCrane(crane_id=crane_id, missing=missing)
            for crane_id, missing in zone.held_cranes.items()
        ]


def release_seq_now(zone: ZoneState) -> int:
    with zone._lock:  # noqa: SLF001
        return zone.release_seq


def occupancy_version(zone: ZoneState) -> int:
    with zone._lock:  # noqa: SLF001
        return zone.occupancy_version


def persist(zone: ZoneState, store: Store) -> None:
    """Write the zone snapshot and mark the zone durable.

    The revision is bumped only after the store accepts the payload, so a
    failed write leaves the zone non-durable and blocks any swing.
    """

    with zone._lock:  # noqa: SLF001
        snapshot = {
            "id": zone.id,
            "name": zone.name,
            "boundary": [point.to_dict() for point in zone.boundary],
            "revision": zone.revision,
            "occupants": list(zone.occupants),
            "registrations": list(zone.registrations),
            "conflict": zone.conflict,
            "latch": zone.latch,
            "held_cranes": {
                crane_id: list(missing) for crane_id, missing in zone.held_cranes.items()
            },
            "release_ledger": [entry.to_dict() for entry in zone.release_ledger],
            "release_seq": zone.release_seq,
        }
        payload = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
        store.save(_zone_key(zone.id), payload)
        zone.revision += 1
        zone.durable = True


def restore(zone: ZoneState, store: Store) -> None:
    """Reload the newest persisted zone snapshot on boot."""

    with zone._lock:  # noqa: SLF001
        try:
            payload = store.load(_zone_key(zone.id))
        except StoreNotFound:
            zone.durable = False
            return
        snapshot = json.loads(payload.decode("utf-8"))
        zone.name = snapshot["name"]
        zone.boundary = [Point(point["x"], point["y"]) for point in snapshot["boundary"]]
        zone.revision = snapshot["revision"]
        zone.occupants = {crane_id: True for crane_id in snapshot["occupants"]}
        zone.registrations = list(snapshot["registrations"])
        zone.conflict = snapshot["conflict"]
        zone.latch = snapshot["latch"]
        zone.held_cranes = {
            str(crane_id): tuple(missing)
            for crane_id, missing in snapshot.get("held_cranes", {}).items()
        }
        zone.release_ledger = [
            ReleaseEntry(
                crane_id=str(entry["crane_id"]),
                generation=int(entry["generation"]),
                seq=int(entry.get("seq", 0)),
                basis=str(entry.get("basis", "")),
            )
            for entry in snapshot.get("release_ledger", [])
        ]
        zone.release_seq = int(snapshot.get("release_seq", 0))
        zone.durable = True


def detect_conflicts(zone: ZoneState, positions: Mapping[str, Point], safe: float) -> list[Conflict]:
    """Return every pair of occupying cranes closer than ``safe`` metres."""

    with zone._lock:  # noqa: SLF001
        cranes = sorted(zone.occupants)
    conflicts: list[Conflict] = []
    for index, first in enumerate(cranes):
        for second in cranes[index + 1 :]:
            point_a = positions.get(first)
            point_b = positions.get(second)
            if point_a is None or point_b is None:
                continue
            if point_a.distance(point_b) < safe:
                conflicts.append(Conflict(zone_id=zone.id, crane_ids=(first, second)))
    return conflicts


def register_conflict(zone: ZoneState, crane_ids: Sequence[str]) -> None:
    """Raise the conflict flag, latch the zone and record the pair."""

    with zone._lock:  # noqa: SLF001
        zone.conflict = True
        zone.latch = True
        for crane_id in crane_ids:
            if crane_id:
                zone.occupants[crane_id] = True
                zone.conflict_pairs.append(crane_id)
                zone.held_cranes.setdefault(crane_id, (HELD_CONFLICT,))


def lock_zone(zone: ZoneState) -> None:
    """Advance the latch generation and hold every crane inside the zone."""

    with zone._lock:  # noqa: SLF001
        zone.latch = True
        zone.latch_generation += 1
        for crane_id in list(zone.conflict_pairs) + list(zone.occupants):
            zone.held_cranes.setdefault(crane_id, (HELD_CONFLICT,))


def _release_candidates(zone: ZoneState) -> list[str]:
    """Order the clear candidates by their zone registration sequence."""

    pool = set(zone.held_cranes) | set(zone.conflict_pairs) | set(zone.occupants)

    def sort_key(crane_id: str) -> tuple[bool, int, str]:
        seq = zone.registration_seq_by_crane.get(crane_id, 0)
        return (seq == 0, seq, crane_id)

    return sorted(pool, key=sort_key)


def _assess_release(
    zone: ZoneState, crane_id: str, positions: Mapping[str, Point], safe: float
) -> tuple[bool, str, list[str]]:
    """Decide whether one crane may leave the latch, and why (not)."""

    if crane_id not in zone.occupants:
        return True, BASIS_VACATED, []
    own = positions.get(crane_id)
    if own is None:
        return False, "", [MISSING_OWN_POSITION]
    missing: list[str] = []
    nearest: float | None = None
    for other in sorted(zone.occupants):
        if other == crane_id:
            continue
        point = positions.get(other)
        if point is None:
            missing.append(f"缺少{other}的实时位置，无法核验间距")
            continue
        distance = own.distance(point)
        if nearest is None or distance < nearest:
            nearest = distance
        if distance < safe:
            missing.append(f"与{other}间距{distance:.1f}m，小于安全距离{safe:.1f}m")
    if missing:
        return False, "", missing
    if nearest is None:
        return True, BASIS_ALONE, []
    return True, f"与最近车辆间距{nearest:.1f}m，不小于安全距离{safe:.1f}m", []


def clear_conflict(
    zone: ZoneState,
    notifier: ConflictNotifier | None = None,
    positions: Mapping[str, Point] | None = None,
    safe: float = 0.0,
) -> ReleaseOutcome:
    """Release the latch crane by crane and record the order and basis.

    Every held crane, recorded conflict pair and current occupant is
    assessed on its own: a crane is released once its occupancy has exited
    the zone, or once its clearance to every remaining occupant checks out
    against the live positions. Cranes that fall short stay latched with
    the missing conditions noted, and the zone latch holds until the last
    crane is released. Returns the outcome so callers can correlate the
    notification with the release ledger.
    """

    live = positions or {}
    with zone._lock:  # noqa: SLF001
        zone.latch_generation += 1
        generation = zone.latch_generation
        candidates = _release_candidates(zone)
        conflicts = list(zone.conflict_pairs)
        zone.conflict_pairs = []
        releases: list[ReleaseEntry] = []
        held: dict[str, tuple[str, ...]] = {}
        for crane_id in candidates:
            releasable, basis, missing = _assess_release(zone, crane_id, live, safe)
            if releasable:
                zone.release_seq += 1
                releases.append(
                    ReleaseEntry(
                        crane_id=crane_id,
                        generation=generation,
                        seq=zone.release_seq,
                        basis=basis,
                    )
                )
            else:
                held[crane_id] = tuple(missing)
        zone.held_cranes = held
        zone.release_ledger.extend(releases)
        zone.conflict = bool(held)
        zone.latch = bool(held)
    if notifier is not None and releases:
        notifier.on_conflict_cleared(
            zone.id, [entry.crane_id for entry in releases], generation, conflicts
        )
    return ReleaseOutcome(
        generation=generation,
        releases=tuple(releases),
        held=tuple(
            HeldCrane(crane_id=crane_id, missing=missing)
            for crane_id, missing in held.items()
        ),
    )


def coverage_check(zone: ZoneState, positions: Mapping[str, Point]) -> Coverage:
    """Split the supplied crane positions into inside/outside groups."""

    with zone._lock:  # noqa: SLF001
        boundary = list(zone.boundary)
    polygon = Polygon.from_points(boundary)
    bounds = polygon.bounds()
    expanded = polygon.expand(2.0)
    inside: list[str] = []
    outside: list[str] = []
    inside_bounds: list[str] = []
    for crane_id, point in positions.items():
        if polygon.contains(point):
            inside.append(crane_id)
        else:
            outside.append(crane_id)
        if bounds.contains(point):
            inside_bounds.append(crane_id)
    return Coverage(
        inside=tuple(sorted(inside)),
        outside=tuple(sorted(outside)),
        inside_bounds=tuple(sorted(inside_bounds)),
        area=polygon.area(),
        bounds_width=bounds.width(),
        bounds_height=bounds.height(),
        center=bounds.center(),
        expanded_area=expanded.area(),
        expanded_overlap=expanded.bounds().overlaps(bounds),
    )


def boundary_area(zone: ZoneState) -> float:
    with zone._lock:  # noqa: SLF001
        boundary = list(zone.boundary)
    return Polygon.from_points(boundary).area()


def separation(zone: ZoneState, positions: Mapping[str, Point]) -> DistanceReport:
    """Compute the closest clearance between every occupying crane pair."""

    with zone._lock:  # noqa: SLF001
        revision = zone.revision
        zone.cached_positions = {key: point for key, point in positions.items()}
    cached = {key: point for key, point in positions.items()}
    crane_ids = occupants_snapshot(zone)
    closest = -1.0
    if len(crane_ids) >= 2:
        for index, first in enumerate(crane_ids):
            for second in crane_ids[index + 1 :]:
                point_a = cached.get(first)
                point_b = cached.get(second)
                if point_a is None or point_b is None:
                    continue
                distance = point_a.distance(point_b)
                if closest < 0 or distance < closest:
                    closest = distance
    return DistanceReport(distance=closest, positions=cached, revision=revision, fresh=True)
