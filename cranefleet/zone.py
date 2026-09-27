"""Anti-collision zone state.

Every zone owns the occupancy registrations reported by the cranes working
inside it, the conflict flag raised when two cranes get too close, and the
per-crane latches that keep each swing blocked until that crane is safe to
release. A latch is never released fleet-wide: when the conflict clears the
zone evaluates every latched crane on its own merits (occupancy exited,
fresh surveyed position, safe separation from every peer) and releases only
the cranes that pass, keeping the rest locked with an explicit reason. The
release ledger records the order and basis of every release. The zone is the
component whose state must be durable before any swing is allowed, so the
persistence helpers live here rather than in the console.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .geometry import Point, Polygon
from .errors import StoreNotFound
from .store import Store

# Release basis recorded for every released crane.
BASIS_POSITION_KNOWN = "position-known"
BASIS_OCCUPANCY_CLEARED = "occupancy-cleared"
BASIS_SAFE_SEPARATION = "safe-separation"
RELEASE_BASIS = (
    BASIS_POSITION_KNOWN,
    BASIS_OCCUPANCY_CLEARED,
    BASIS_SAFE_SEPARATION,
)

# Withhold reasons explaining why a crane stays latched.
REASON_MISSING_POSITION = "missing-position"
REASON_STILL_OCCUPYING = "still-occupying"
REASON_PEER_UNLOCATED = "peer-unlocated"


def reason_too_close(crane_id: str) -> str:
    """Withhold reason: the crane is closer than the safe distance to a peer."""

    return f"too-close:{crane_id}"


def reason_peer_unlocated(crane_id: str) -> str:
    """Withhold reason: a peer has no fresh position so safety cannot be proven."""

    return f"{REASON_PEER_UNLOCATED}:{crane_id}"


@dataclass(frozen=True)
class ReleaseEntry:
    """One crane that was released from the latch.

    ``order`` is the zone-wide 1-based release sequence and ``latch_seq`` the
    order in which the crane was latched, so the ledger shows the queue
    discipline. ``basis`` lists the checks that passed and ``min_distance`` is
    the closest clearance measured at release time.
    """

    order: int
    crane_id: str
    generation: int
    basis: tuple[str, ...] = field(default=())
    min_distance: float = -1.0
    latch_seq: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "crane_id": self.crane_id,
            "generation": self.generation,
            "basis": list(self.basis),
            "min_distance": self.min_distance,
            "latch_seq": self.latch_seq,
        }


@dataclass(frozen=True)
class WithheldEntry:
    """A latched crane that failed evaluation, with every missing condition."""

    crane_id: str
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    min_distance: float
    safe_distance: float
    occupying: bool
    position_known: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "crane_id": self.crane_id,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "min_distance": self.min_distance,
            "safe_distance": self.safe_distance,
            "occupying": self.occupying,
            "position_known": self.position_known,
        }


@dataclass(frozen=True)
class ReleaseDecision:
    """Outcome of one release evaluation pass."""

    generation: int
    released: tuple[ReleaseEntry, ...]
    withheld: tuple[WithheldEntry, ...]

    def released_ids(self) -> list[str]:
        return [entry.crane_id for entry in self.released]

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "released": [entry.to_dict() for entry in self.released],
            "withheld": [entry.to_dict() for entry in self.withheld],
        }


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


class LatchReleaseNotifier(Protocol):
    """Receives one callback per crane released from a zone latch."""

    def on_latch_released(self, zone_id: str, crane_id: str, generation: int) -> None:
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
        # crane_id -> sequence in which it was latched (queue discipline)
        self.latched: dict[str, int] = {}
        self.latch_seq = 0
        self.release_seq = 0
        self.cached_positions: dict[str, Point] = {}
        self.registration_seq = 0
        self.registration_seq_by_crane: dict[str, int] = {}
        self.release_ledger: list[ReleaseEntry] = []
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
    """Remove a crane from the zone once it leaves.

    The crane is only dropped from the occupancy registrations; a latched
    crane keeps its latch until ``evaluate_releases`` proves it is safe.
    """

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
    """True while at least one crane is still held by the zone latch."""

    with zone._lock:  # noqa: SLF001
        return bool(zone.latched)


def crane_latched(zone: ZoneState, crane_id: str) -> bool:
    with zone._lock:  # noqa: SLF001
        return crane_id in zone.latched


def latched_cranes(zone: ZoneState) -> list[str]:
    """Latched cranes in latch order (first latched first)."""

    with zone._lock:  # noqa: SLF001
        return [crane_id for crane_id, _ in sorted(zone.latched.items(), key=_latch_key)]


def latch_generation(zone: ZoneState) -> int:
    with zone._lock:  # noqa: SLF001
        return zone.latch_generation


def release_ledger_of(zone: ZoneState) -> list[ReleaseEntry]:
    with zone._lock:  # noqa: SLF001
        return list(zone.release_ledger)


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
            "latched": dict(zone.latched),
            "latch_seq": zone.latch_seq,
            "release_seq": zone.release_seq,
            "latch_generation": zone.latch_generation,
            "release_ledger": [entry.to_dict() for entry in zone.release_ledger],
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
        zone.latch_seq = snapshot.get("latch_seq", 0)
        zone.release_seq = snapshot.get("release_seq", 0)
        zone.latch_generation = snapshot.get("latch_generation", 0)
        zone.release_ledger = [
            ReleaseEntry(
                order=item["order"],
                crane_id=item["crane_id"],
                generation=item["generation"],
                basis=tuple(item.get("basis", ())),
                min_distance=item.get("min_distance", -1.0),
                latch_seq=item.get("latch_seq", 0),
            )
            for item in snapshot.get("release_ledger", [])
        ]
        latched = snapshot.get("latched")
        if latched is None:
            # Backwards compatibility with snapshots written before per-crane
            # latches: a held latch applied to every occupant.
            latched = {}
            if snapshot["latch"]:
                for crane_id in sorted(zone.occupants):
                    zone.latch_seq += 1
                    latched[crane_id] = zone.latch_seq
        zone.latched = dict(latched)
        zone.latch = bool(zone.latched)
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
    """Raise the conflict flag and latch each crane involved.

    Cranes already latched keep their original latch sequence so the release
    queue order is stable across repeated conflict reports.
    """

    with zone._lock:  # noqa: SLF001
        zone.conflict = True
        zone.latch = True
        for crane_id in crane_ids:
            if not crane_id:
                continue
            zone.occupants.setdefault(crane_id, True)
            if crane_id not in zone.latched:
                zone.latch_seq += 1
                zone.latched[crane_id] = zone.latch_seq


def _latch_key(item: tuple[str, int]) -> tuple[int, str]:
    crane_id, sequence = item
    return sequence, crane_id


def _assess_crane(
    crane_id: str,
    occupants: frozenset[str],
    latched_peers: frozenset[str],
    positions: Mapping[str, Point],
    safe: float,
) -> tuple[list[str], list[str], float, bool, bool]:
    """Evaluate one crane against the release conditions.

    Returns the missing-condition reasons, the blocking peers, the closest
    measured clearance (``-1`` when unknown) and the occupancy/position flags.
    The peer set is fixed by the caller so two cranes vacating together can
    never release each other in the same pass.
    """

    reasons: list[str] = []
    point = positions.get(crane_id)
    position_known = point is not None
    occupying = crane_id in occupants
    if not position_known:
        reasons.append(REASON_MISSING_POSITION)
    if occupying:
        reasons.append(REASON_STILL_OCCUPYING)
    blockers: list[str] = []
    min_distance = -1.0
    peers = sorted((occupants | latched_peers) - {crane_id})
    for peer in peers:
        peer_point = positions.get(peer)
        if peer_point is None:
            reasons.append(reason_peer_unlocated(peer))
            continue
        if point is None:
            continue
        distance = point.distance(peer_point)
        if min_distance < 0.0 or distance < min_distance:
            min_distance = distance
        if distance < safe:
            blockers.append(peer)
            reasons.append(reason_too_close(peer))
    return reasons, blockers, min_distance, occupying, position_known


def _withheld_from_assessment(
    crane_id: str,
    reasons: Sequence[str],
    blockers: Sequence[str],
    min_distance: float,
    safe: float,
    occupying: bool,
    position_known: bool,
) -> WithheldEntry:
    return WithheldEntry(
        crane_id=crane_id,
        reasons=tuple(dict.fromkeys(reasons)),
        blockers=tuple(blockers),
        min_distance=min_distance,
        safe_distance=safe,
        occupying=occupying,
        position_known=position_known,
    )


def evaluate_releases(
    zone: ZoneState,
    positions: Mapping[str, Point],
    safe: float,
    notifier: LatchReleaseNotifier | None = None,
) -> ReleaseDecision:
    """Release every latched crane that currently meets all conditions.

    Each latched crane is evaluated on its own. A crane is released only when
    it has a fresh position, its own occupancy registration has exited, and it
    keeps at least ``safe`` metres from every crane still occupying the zone
    or still latched. Eligible cranes are released in latch order; the others
    stay locked and are returned with the conditions they are missing. The
    generation advances once per pass that releases at least one crane.
    """

    with zone._lock:  # noqa: SLF001
        candidates = [crane_id for crane_id, _ in sorted(zone.latched.items(), key=_latch_key)]
        occupants = frozenset(zone.occupants)
        latched_peers = frozenset(zone.latched)
        released: list[ReleaseEntry] = []
        withheld: list[WithheldEntry] = []
        generation = zone.latch_generation
        for crane_id in candidates:
            reasons, blockers, min_distance, occupying, position_known = _assess_crane(
                crane_id, occupants, latched_peers, positions, safe
            )
            if reasons:
                withheld.append(
                    _withheld_from_assessment(
                        crane_id, reasons, blockers, min_distance, safe, occupying, position_known
                    )
                )
                continue
            if not released:
                zone.latch_generation += 1
                generation = zone.latch_generation
            zone.release_seq += 1
            latch_seq = zone.latched.pop(crane_id)
            entry = ReleaseEntry(
                order=zone.release_seq,
                crane_id=crane_id,
                generation=generation,
                basis=RELEASE_BASIS,
                min_distance=min_distance,
                latch_seq=latch_seq,
            )
            zone.release_ledger.append(entry)
            released.append(entry)
        zone.conflict = bool(_occupant_conflicts(occupants, positions, safe))
        zone.latch = bool(zone.latched)
        decision = ReleaseDecision(
            generation=generation, released=tuple(released), withheld=tuple(withheld)
        )
    if notifier is not None:
        for entry in released:
            notifier.on_latch_released(zone.id, entry.crane_id, entry.generation)
    return decision


def _occupant_conflicts(
    occupants: frozenset[str], positions: Mapping[str, Point], safe: float
) -> list[tuple[str, str]]:
    cranes = sorted(occupants)
    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(cranes):
        for second in cranes[index + 1 :]:
            point_a = positions.get(first)
            point_b = positions.get(second)
            if point_a is None or point_b is None:
                continue
            if point_a.distance(point_b) < safe:
                pairs.append((first, second))
    return pairs


def latch_report(zone: ZoneState, positions: Mapping[str, Point], safe: float) -> list[WithheldEntry]:
    """Explain why every currently latched crane is still held.

    The report is a live evaluation, so a crane that has since become safe is
    returned with empty reasons: it is waiting for the next release pass.
    """

    with zone._lock:  # noqa: SLF001
        candidates = [crane_id for crane_id, _ in sorted(zone.latched.items(), key=_latch_key)]
        occupants = frozenset(zone.occupants)
        latched_peers = frozenset(zone.latched)
    entries: list[WithheldEntry] = []
    for crane_id in candidates:
        reasons, blockers, min_distance, occupying, position_known = _assess_crane(
            crane_id, occupants, latched_peers, positions, safe
        )
        entries.append(
            _withheld_from_assessment(
                crane_id, reasons, blockers, min_distance, safe, occupying, position_known
            )
        )
    return entries


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
