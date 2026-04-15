"""
Zone-based event detection: intrusion and loitering.

Design
------
* Zones are defined as 2-D polygons loaded from zones.json.
* The "ground position" of a person is their bounding-box bottom-centre
  point, smoothed over the last `smoothing_window` frames to remove jitter.
* A person loiters when they remain in a zone AND their displacement from
  the zone-entry anchor is below `loitering_movement_threshold_pixels`.
* Each event type fires at most once per (person, zone, entry) to prevent
  log spam; a configurable cooldown window controls re-alerting.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

from .config import PipelineCfg
from .types import PersonState, TrackInfo, TrackedResult, ZoneEvent

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════ #
#  Zone definition
# ══════════════════════════════════════════════════════════════════════════ #

@dataclass
class Zone:
    name: str
    polygon: Polygon          # Shapely; built from polygon_2d coordinates
    loitering_threshold: float
    zone_type: str            # "restricted" | "monitored" | "entry_exit"
    color_bgr: Tuple[int, int, int] = (0, 0, 255)
    raw_pts: List[Tuple[int, int]] = field(default_factory=list)


def load_zones(path: str) -> List[Zone]:
    """Parse zones.json and return a list of Zone objects."""
    p = Path(path)
    if not p.exists():
        logger.warning("zones.json not found at %s — no zones active", path)
        return []

    with p.open() as fh:
        data = json.load(fh)

    zones: List[Zone] = []
    for entry in data.get("zones", []):
        pts_2d = entry.get("polygon_2d", [])
        if len(pts_2d) < 3:
            logger.warning("Zone %r has < 3 points — skipped", entry.get("name"))
            continue
        poly = Polygon(pts_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)  # auto-repair self-intersections

        color_hex = entry.get("color", "#FF0000").lstrip("#")
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)

        zones.append(Zone(
            name=entry.get("name", f"Zone_{len(zones)}"),
            polygon=poly,
            loitering_threshold=float(entry.get("loitering_threshold_seconds", 10.0)),
            zone_type=entry.get("type", "restricted"),
            color_bgr=(b, g, r),          # OpenCV uses BGR
            raw_pts=[(int(x), int(y)) for x, y in pts_2d],
        ))

    logger.info("Loaded %d zone(s) from %s", len(zones), path)
    return zones


# ══════════════════════════════════════════════════════════════════════════ #
#  Zone logic engine
# ══════════════════════════════════════════════════════════════════════════ #

class ZoneLogic:
    """
    Stateful engine that processes one TrackedResult per call and returns
    zone events plus an updated person-state table.

    State persists across frames inside this object.
    """

    def __init__(self, cfg: PipelineCfg, zones: List[Zone]) -> None:
        self._cfg = cfg
        self._zones = zones
        # session_id → PersonState
        self._states: Dict[int, PersonState] = {}

    # ── public API ────────────────────────────────────────────────────── #

    def process(
        self,
        result: TrackedResult,
    ) -> Tuple[List[ZoneEvent], Dict[int, str], Dict[int, str]]:
        """
        Parameters
        ----------
        result : TrackedResult

        Returns
        -------
        zone_events     : events fired this frame
        activities      : session_id → activity label
        active_alerts   : session_id → alert overlay text
        """
        ts = result.raw.timestamp
        zone_events: List[ZoneEvent] = []
        activities: Dict[int, str] = {}
        active_alerts: Dict[int, str] = {}

        active_sids = {t.session_id for t in result.tracks}

        # ── Remove state for tracks that have disappeared ── #
        for sid in list(self._states.keys()):
            if sid not in active_sids:
                del self._states[sid]

        for track in result.tracks:
            sid = track.session_id
            if sid not in self._states:
                self._states[sid] = PersonState(
                    permanent_id=track.permanent_id,
                    session_id=sid,
                )
            state = self._states[sid]
            state.permanent_id = track.permanent_id   # may be updated after re-ID

            # ── Compute smoothed ground point ── #
            gx, gy = self._ground_point(track.bbox)
            state.position_history.append((gx, gy))
            if len(state.position_history) > self._cfg.zone.smoothing_window:
                state.position_history.pop(0)
            sgx, sgy = self._smooth(state.position_history)

            # ── Activity from motion ── #
            activity = self._compute_activity(state, sgx, sgy, result.raw.fps)
            state.activity = activity
            activities[sid] = activity

            # ── Zone membership ── #
            point = Point(sgx, sgy)
            matched_zone: Optional[Zone] = None
            for zone in self._zones:
                if zone.polygon.contains(point):
                    matched_zone = zone
                    break

            if matched_zone is None:
                # ── Person left zone ── #
                if state.in_zone:
                    logger.debug(
                        "Person %s (session %d) left %s",
                        state.permanent_id, sid, state.current_zone,
                    )
                    state.in_zone = False
                    state.current_zone = None
                    state.loitering_alerted = False
                    state.intrusion_alerted = False
            else:
                # ── Person inside zone ── #
                if not state.in_zone:
                    # Fresh entry
                    state.in_zone = True
                    state.current_zone = matched_zone.name
                    state.zone_entry_time = ts
                    state.anchor_x = sgx
                    state.anchor_y = sgy
                    state.loitering_alerted = False
                    state.intrusion_alerted = False

                # ── Intrusion event ── #
                cooldown = self._cfg.zone.alert_cooldown_seconds
                if not state.intrusion_alerted or (
                    ts - state.last_intrusion_time > cooldown
                ):
                    ev = ZoneEvent(
                        event_type="ZONE_INTRUSION",
                        zone_name=matched_zone.name,
                        permanent_id=state.permanent_id,
                        session_id=sid,
                        bbox=track.bbox.copy(),
                        confidence=track.confidence,
                    )
                    zone_events.append(ev)
                    active_alerts[sid] = f"INTRUSION: {matched_zone.name}"
                    state.intrusion_alerted = True
                    state.last_intrusion_time = ts
                elif state.intrusion_alerted:
                    active_alerts[sid] = f"IN ZONE: {matched_zone.name}"

                # ── Loitering check ── #
                displacement = np.hypot(sgx - state.anchor_x, sgy - state.anchor_y)
                dwell = ts - state.zone_entry_time
                move_thresh = self._cfg.zone.loitering_movement_threshold_pixels

                if displacement > move_thresh:
                    # Person moved significantly — reset anchor
                    state.anchor_x = sgx
                    state.anchor_y = sgy
                    state.zone_entry_time = ts
                    state.loitering_alerted = False
                elif (
                    dwell >= matched_zone.loitering_threshold
                    and not state.loitering_alerted
                ):
                    ev = ZoneEvent(
                        event_type="LOITERING_DETECTED",
                        zone_name=matched_zone.name,
                        permanent_id=state.permanent_id,
                        session_id=sid,
                        bbox=track.bbox.copy(),
                        confidence=track.confidence,
                        duration_seconds=dwell,
                    )
                    zone_events.append(ev)
                    active_alerts[sid] = f"LOITERING: {matched_zone.name} ({dwell:.0f}s)"
                    state.loitering_alerted = True
                    state.last_loitering_time = ts

        return zone_events, activities, active_alerts

    # ── static helpers ────────────────────────────────────────────────── #

    @staticmethod
    def _ground_point(bbox: np.ndarray) -> Tuple[float, float]:
        """Bottom-centre of the bounding box = person's standing position."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, float(y2)

    @staticmethod
    def _smooth(history: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not history:
            return 0.0, 0.0
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        return float(np.mean(xs)), float(np.mean(ys))

    def _compute_activity(
        self,
        state: PersonState,
        cx: float,
        cy: float,
        fps: float,
    ) -> str:
        """
        Motion-based activity heuristic (pixels/second).
        CLIP overrides this for semantic understanding when needed.
        """
        if len(state.position_history) < 2:
            return "Person is standing"

        prev_x, prev_y = state.position_history[-2]
        dx = cx - prev_x
        dy = cy - prev_y
        # displacement per second
        speed = np.hypot(dx, dy) * max(fps, 1.0)

        standing = self._cfg.anomaly.activity_motion_standing_px_s
        running = self._cfg.anomaly.activity_motion_running_px_s

        if speed < standing:
            return "Person is standing"
        if speed >= running:
            return "Person is running"
        return "Person is walking"