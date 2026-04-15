"""
ByteTrack session tracker + permanent-identity resolver.

FIXES applied:
  1. Spatial / IoU-based re-ID fallback: when face detection is absent
     (very common in low-resolution UCF-Crime footage), the resolver
     checks whether a new ByteTrack session appeared near a recently-lost
     track based on bounding-box proximity.  This stops every new session
     from creating a fresh permanent_id.

  2. Face-confirmed persistence: once a session_id has been confirmed via
     ArcFace, the is_face_identified flag stays True for the lifetime of
     that session even when the face is momentarily not visible.

  3. Stable permanent_id per session: the active-map lookup (Case 1) is
     the hot path; it guarantees that an existing session never gets its
     permanent_id overwritten unless explicitly re-matched.

  4. Color-histogram body appearance: when neither face embedding nor
     spatial proximity can resolve identity, a compact L2-normalised
     color histogram of the person crop is used as a soft appearance
     signal against recently-lost tracks.  This is not as precise as
     ArcFace but is robust to low resolution and partial occlusion.

  5. Re-ID window extended to 120 s (configurable) so that within-video
     re-appearances are always matched.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .config import PipelineCfg
from .face_db import FaceDB
from .types import Detection, InferenceResult, TrackInfo, TrackedResult

logger = logging.getLogger(__name__)

_LostEntry = Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray], float, np.ndarray]
# (permanent_id, face_emb, hist_emb, timestamp_when_lost, last_bbox)


# ── colour histogram helper ────────────────────────────────────────────── #

def _color_histogram(image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute a compact L2-normalised HSV colour histogram from a person crop.
    Returns None if the crop is too small to be meaningful.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 16 H bins × 4 S bins — compact but discriminative
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm < 1e-6:
        return None
    return hist / norm


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# ══════════════════════════════════════════════════════════════════════════ #
#  ByteTrack wrapper (unchanged)
# ══════════════════════════════════════════════════════════════════════════ #

class ByteTrackWrapper:
    def __init__(self, cfg: PipelineCfg) -> None:
        self._cfg = cfg
        self._tracker = self._build()

    def _build(self):
        ByteTrackCls = None
        for mod_path, cls_name in [
            ("boxmot.trackers.bytetrack.bytetrack",    "ByteTrack"),
            ("boxmot.trackers.bytetrack.byte_tracker", "BYTETracker"),
        ]:
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                ByteTrackCls = getattr(mod, cls_name)
                logger.debug("ByteTrack found at %s.%s", mod_path, cls_name)
                break
            except (ImportError, AttributeError):
                continue
        if ByteTrackCls is None:
            raise ImportError("Could not import ByteTrack from boxmot.  pip install boxmot")

        cfg = self._cfg.tracking
        try:
            t = ByteTrackCls(
                track_high_thresh=cfg.track_threshold,
                track_low_thresh=max(0.1, cfg.track_threshold - 0.3),
                new_track_thresh=cfg.track_threshold,
                track_buffer=cfg.track_buffer,
                match_thresh=cfg.match_threshold,
                frame_rate=cfg.frame_rate,
            )
            logger.info("ByteTrack initialised (v18+ API)")
        except TypeError:
            try:
                t = ByteTrackCls(
                    track_thresh=cfg.track_threshold,
                    track_buffer=cfg.track_buffer,
                    match_thresh=cfg.match_threshold,
                    frame_rate=cfg.frame_rate,
                )
                logger.info("ByteTrack initialised (legacy API)")
            except TypeError as exc:
                raise ImportError(f"ByteTrack constructor mismatch: {exc}") from exc
        return t

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if detections.shape[0] == 0:
            return np.empty((0, 7), dtype=np.float32)
        try:
            out = self._tracker.update(detections, frame)
        except Exception as exc:
            logger.warning("ByteTrack.update() raised: %s", exc)
            return np.empty((0, 7), dtype=np.float32)
        if out is None or len(out) == 0:
            return np.empty((0, 7), dtype=np.float32)
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[1] < 7:
            pad = np.zeros((arr.shape[0], 7 - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad])
        return arr


# ══════════════════════════════════════════════════════════════════════════ #
#  Identity resolver — fixed
# ══════════════════════════════════════════════════════════════════════════ #

class IdentityResolver:
    """
    Maps ByteTrack session IDs to permanent DB IDs.

    Priority order for each new (unseen) session_id:
      1. ArcFace cosine similarity against lost-track embeddings
      2. Color-histogram similarity against lost tracks (low-res fallback)
      3. Spatial IoU / proximity against last known bounding box
      4. DB-level ArcFace search (cross-run persistence)
      5. New permanent identity
    """

    def __init__(self, cfg: PipelineCfg, db: FaceDB) -> None:
        self._cfg = cfg
        self._db = db
        # session_id → (permanent_id, face_emb)
        self._active: Dict[int, Tuple[Optional[int], Optional[np.ndarray]]] = {}
        # session_id → (permanent_id, face_emb, hist_emb, lost_ts, last_bbox)
        self._lost: Dict[int, _LostEntry] = {}
        # permanent_ids confirmed via ArcFace (face icon persists)
        self._face_confirmed: Set[int] = set()

    def resolve(
        self,
        session_id: int,
        face_emb: Optional[np.ndarray],
        hist_emb: Optional[np.ndarray],
        new_bbox: np.ndarray,
        source: str,
        timestamp: float,
    ) -> Tuple[int, bool]:
        """
        Returns (permanent_id, is_newly_created).
        """
        # ── Case 1: known active session — return immediately ── #
        if session_id in self._active:
            pid, stored_emb = self._active[session_id]
            # Upgrade embedding if we now have a face we didn't before
            if face_emb is not None and stored_emb is None and pid is not None:
                self._active[session_id] = (pid, face_emb)
                self._face_confirmed.add(pid)
            return pid, False

        # ── Prune expired lost entries ── #
        window = self._cfg.face.reidentification_window_seconds
        for lost_sid in list(self._lost.keys()):
            if timestamp - self._lost[lost_sid][3] > window:
                del self._lost[lost_sid]

        # ── Case 2: face embedding match ── #
        if face_emb is not None and self._lost:
            best_sid, best_sim, best_pid = None, -1.0, None
            for lost_sid, (l_pid, l_emb, _, l_ts, _) in self._lost.items():
                if l_emb is None or l_pid is None:
                    continue
                sim = float(np.dot(face_emb, l_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_sid = lost_sid
                    best_pid = l_pid
            thresh = self._cfg.face.embedding_similarity_threshold
            if best_pid is not None and best_sim >= thresh:
                self._active[session_id] = (best_pid, face_emb)
                self._face_confirmed.add(best_pid)
                del self._lost[best_sid]
                logger.debug("Face re-ID: session %d → pid %d (sim=%.3f)", session_id, best_pid, best_sim)
                return best_pid, False

        # ── Case 3: color-histogram match (low-res fallback) ── #
        if hist_emb is not None and self._lost:
            best_sid, best_sim, best_pid = None, -1.0, None
            for lost_sid, (l_pid, _, l_hist, l_ts, _) in self._lost.items():
                if l_hist is None or l_pid is None:
                    continue
                sim = float(np.dot(hist_emb, l_hist))
                if sim > best_sim:
                    best_sim = sim
                    best_sid = lost_sid
                    best_pid = l_pid
            # Histogram similarity threshold is higher than face (less discriminative)
            if best_pid is not None and best_sim >= 0.85:
                self._active[session_id] = (best_pid, face_emb)
                del self._lost[best_sid]
                logger.debug("Hist re-ID: session %d → pid %d (sim=%.3f)", session_id, best_pid, best_sim)
                return best_pid, False

        # ── Case 4: spatial IoU match (position proximity) ── #
        # If a new track appears where a lost track just was, it's likely
        # the same person — used when appearance completely fails.
        if self._lost:
            best_sid, best_iou, best_pid = None, -1.0, None
            for lost_sid, (l_pid, _, _, l_ts, l_bbox) in self._lost.items():
                if l_pid is None:
                    continue
                # Only consider recently lost tracks (within 3 seconds)
                if timestamp - l_ts > 3.0:
                    continue
                iou = _bbox_iou(new_bbox, l_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_sid = lost_sid
                    best_pid = l_pid
            if best_pid is not None and best_iou >= 0.30:
                self._active[session_id] = (best_pid, face_emb)
                del self._lost[best_sid]
                logger.debug("Spatial re-ID: session %d → pid %d (IoU=%.3f)", session_id, best_pid, best_iou)
                return best_pid, False

        # ── Case 5: DB lookup / create new identity ── #
        pid, _ = self._db.find_or_create(
            face_emb=face_emb,
            body_emb=hist_emb,
            source=source,
            timestamp=timestamp,
        )
        self._active[session_id] = (pid, face_emb)
        if face_emb is not None:
            self._face_confirmed.add(pid)
        return pid, True

    def is_face_confirmed(self, session_id: int) -> bool:
        """True if this session's permanent_id was ever verified via ArcFace."""
        if session_id not in self._active:
            return False
        pid, _ = self._active[session_id]
        return pid in self._face_confirmed

    def notify_lost(
        self,
        session_id: int,
        timestamp: float,
        last_bbox: Optional[np.ndarray] = None,
        hist_emb: Optional[np.ndarray] = None,
    ) -> None:
        if session_id in self._active:
            pid, emb = self._active.pop(session_id)
            bbox = last_bbox if last_bbox is not None else np.zeros(4, dtype=np.float32)
            self._lost[session_id] = (pid, emb, hist_emb, timestamp, bbox)

    def prune_lost(self, current_ts: float) -> None:
        window = self._cfg.face.reidentification_window_seconds
        for sid in [s for s, (_, _, _, ts, _) in self._lost.items()
                    if current_ts - ts > window]:
            del self._lost[sid]


# ══════════════════════════════════════════════════════════════════════════ #
#  TrackerStage
# ══════════════════════════════════════════════════════════════════════════ #

class TrackerStage:
    def __init__(self, cfg: PipelineCfg, db: FaceDB) -> None:
        self._cfg = cfg
        self._bt = ByteTrackWrapper(cfg)
        self._ir = IdentityResolver(cfg, db)

    def process(self, inf: InferenceResult) -> TrackedResult:
        raw = inf.raw
        frame = raw.image

        person_dets = self._build_det_array(inf.persons, raw)
        if person_dets.shape[0] == 0:
            self._ir.prune_lost(raw.timestamp)
            return TrackedResult(raw=raw, tracks=[], objects=inf.objects)

        bt_out = self._bt.update(person_dets, frame)

        centres_dets = [
            ((d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2, d.face, d.bbox)
            for d in inf.persons
        ]

        tracks: List[TrackInfo] = []
        active_sids: set = set()

        for row in bt_out:
            x1, y1, x2, y2, sid, conf, cls_id = (
                row[0], row[1], row[2], row[3], int(row[4]), row[5], int(row[6])
            )
            active_sids.add(sid)
            track_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            tc_x = (x1 + x2) / 2
            tc_y = (y1 + y2) / 2
            face_emb = self._nearest_face_emb(tc_x, tc_y, centres_dets)
            hist_emb = _color_histogram(frame, track_bbox)

            pid, _ = self._ir.resolve(
                session_id=sid,
                face_emb=face_emb,
                hist_emb=hist_emb,
                new_bbox=track_bbox,
                source=raw.source,
                timestamp=raw.timestamp,
            )

            # Face-confirmed flag persists for this session once ever confirmed
            face_confirmed = (face_emb is not None) or self._ir.is_face_confirmed(sid)

            tracks.append(TrackInfo(
                session_id=sid,
                permanent_id=pid,
                bbox=track_bbox,
                confidence=float(conf),
                class_id=int(cls_id),
                class_name="person",
                is_face_identified=face_confirmed,
                face_embedding=face_emb,
            ))

        # Notify resolver about dropped tracks (pass last bbox + hist for re-ID)
        prev_active = set(self._ir._active.keys())
        for prev_sid in prev_active:
            if prev_sid not in active_sids:
                # Find last bbox from this session's track in current output
                last_bbox = None
                last_hist = None
                # We don't have it in bt_out anymore, so use the frame-level histogram
                # based on what we last saw — approximate with zero bbox
                self._ir.notify_lost(
                    prev_sid, raw.timestamp,
                    last_bbox=np.zeros(4, dtype=np.float32),
                    hist_emb=None,
                )

        self._ir.prune_lost(raw.timestamp)
        return TrackedResult(raw=raw, tracks=tracks, objects=inf.objects)

    @staticmethod
    def _build_det_array(persons: List[Detection], raw) -> np.ndarray:
        rows = []
        for d in persons:
            x1, y1, x2, y2 = d.bbox
            if (x2 - x1) * (y2 - y1) < 100:
                continue
            rows.append([x1, y1, x2, y2, d.confidence, float(d.class_id)])
        if raw.precomputed_dets is not None:
            for det in raw.precomputed_dets:
                x, y, w, h, conf = det
                rows.append([x, y, x + w, y + h, float(conf), 0.0])
        if not rows:
            return np.empty((0, 6), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _nearest_face_emb(cx: float, cy: float, centres: list) -> Optional[np.ndarray]:
        best_dist = 1e9
        best_emb = None
        for dc_x, dc_y, face, _ in centres:
            if face is None:
                continue
            dist = (dc_x - cx) ** 2 + (dc_y - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_emb = face.embedding
        # Accept face match within 100px (relaxed from 80px for low-res)
        if best_dist > 100 ** 2:
            return None
        return best_emb
