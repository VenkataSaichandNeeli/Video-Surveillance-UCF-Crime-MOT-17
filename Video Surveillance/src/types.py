"""
Shared data-transfer objects used across all pipeline stages.

Design rule: every inter-thread payload is an immutable snapshot; no stage
modifies an object produced by a previous stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1 → Stage 2  (Reader → Inference)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RawFrame:
    """A single decoded frame from any input source."""
    frame_idx: int           # 0-based global frame counter
    timestamp: float         # seconds from the start of the source
    image: np.ndarray        # BGR uint8, shape (H, W, 3)
    source: str              # absolute path of the source file / directory
    fps: float               # nominal frame-rate of the source
    # MOT17-only: pre-computed FRCNN detections loaded from det.txt
    # shape (N, 5) → [x, y, w, h, conf]  (top-left origin, MOT format)
    precomputed_dets: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Stage 2 → Stage 3  (Inference → Tracker+Identity)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FaceInfo:
    """A single face detected inside a person crop."""
    bbox: np.ndarray          # [x1, y1, x2, y2] absolute image coords, int32
    confidence: float
    embedding: np.ndarray     # 512-d ArcFace, L2-normalised, float32


@dataclass(slots=True)
class Detection:
    """One object detected by the main detector."""
    bbox: np.ndarray          # [x1, y1, x2, y2] float32
    confidence: float
    class_id: int
    class_name: str
    face: Optional[FaceInfo] = None   # populated for person detections only


@dataclass(slots=True)
class InferenceResult:
    """All detections for one frame after the inference stage."""
    raw: RawFrame
    persons: List[Detection]   # class_id == 0 (person), filtered by conf
    objects: List[Detection]   # all other classes, filtered by conf


# ---------------------------------------------------------------------------
# Stage 3 → Stage 4  (Tracker+Identity → Analytics)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrackInfo:
    """One tracked entity with its resolved identity."""
    session_id: int                     # ByteTrack ID (resets each run)
    permanent_id: Optional[int]         # face-DB row id; None until matched
    bbox: np.ndarray                    # [x1, y1, x2, y2] float32
    confidence: float
    class_id: int
    class_name: str
    is_face_identified: bool = False
    face_embedding: Optional[np.ndarray] = None   # for DB update later


@dataclass(slots=True)
class TrackedResult:
    """Tracking output for one frame."""
    raw: RawFrame
    tracks: List[TrackInfo]     # person tracks with permanent IDs
    objects: List[Detection]    # non-person detections (no tracking IDs)


# ---------------------------------------------------------------------------
# Stage 4 → Stage 5  (Analytics → Output)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ZoneEvent:
    """A zone-triggered event (intrusion or loitering)."""
    event_type: str             # "ZONE_INTRUSION" | "LOITERING_DETECTED"
    zone_name: str
    permanent_id: Optional[int]
    session_id: int
    bbox: np.ndarray
    confidence: float
    duration_seconds: float = 0.0


@dataclass
class AnalyticsResult:
    """Fully-analysed frame ready for output."""
    raw: RawFrame
    tracks: List[TrackInfo]
    objects: List[Detection]
    scene_anomaly: str                    # scene-level CLIP label
    scene_anomaly_confidence: float
    person_activities: Dict[int, str]     # session_id → "Person is walking" etc.
    person_anomalies: Dict[int, str]      # session_id → anomaly label
    zone_events: List[ZoneEvent]
    loggable_events: List[Dict]           # every event row for the CSV
    # session_id → alert text; drives bounding-box colour / overlay text
    active_alerts: Dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Long-lived mutable state kept inside ZoneLogic (not passed between threads)
# ---------------------------------------------------------------------------

@dataclass
class PersonState:
    """Per-person temporal state maintained by the zone-logic module."""
    permanent_id: Optional[int]
    session_id: int
    # ── zone state ──
    in_zone: bool = False
    current_zone: Optional[str] = None
    zone_entry_time: float = 0.0
    anchor_x: float = 0.0
    anchor_y: float = 0.0
    loitering_alerted: bool = False
    intrusion_alerted: bool = False
    last_intrusion_time: float = 0.0
    last_loitering_time: float = 0.0
    # ── motion history for activity & point smoothing ──
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    # ── last known activity ──
    activity: str = "unknown"
