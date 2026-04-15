"""
Surveillance pipeline orchestrator.

Threading model
---------------
                      raw_q          det_q         analytics_q
  ReaderThread  ──────────►  InferenceThread  ──────────►  AnalyticsOutputThread
                (bounded)             (bounded)

  • ReaderThread     : I/O-bound, reads frames from source
  • InferenceThread  : GPU-bound, runs RT-DETR + InsightFace
  • AnalyticsOutputThread : CPU-bound, runs Tracking + ZoneLogic +
                            CLIP + Visualizer + OutputHandler

Sentinel (None) propagates through the queues on natural end or stop().

Graceful shutdown
-----------------
  _stop_event.set()
  → ReaderThread breaks its loop and puts sentinel
  → InferenceThread receives sentinel, flushes, puts sentinel, exits
  → AnalyticsOutputThread receives sentinel, flushes, exits
  → main thread joins all three

Image-folder mode
-----------------
  Tracking is disabled; ZoneLogic and ByteTrack are skipped.
  Each image is processed independently.
"""
from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .anomaly import AnomalyDetector
from .config import PipelineCfg
from .detector import Detector
from .face_db import FaceDB
from .face_detector import FaceDetector
from .ingestion import ReaderThread, open_source
from .mot_eval import MOTEvaluator
from .output_handler import OutputHandler
from .tracker import TrackerStage
from .types import (
    AnalyticsResult,
    Detection,
    InferenceResult,
    RawFrame,
    TrackInfo,
    TrackedResult,
    ZoneEvent,
)
from .visualizer import Visualizer
from .zone_logic import ZoneLogic, load_zones

logger = logging.getLogger(__name__)

_SENTINEL = None
_QUEUE_TIMEOUT = 0.1     # seconds; how long threads wait on empty queues


class SurveillancePipeline:
    """
    End-to-end pipeline.

    Parameters
    ----------
    input_path  : Path to video file, MOT17 sequence dir, or image folder
    output_dir  : Directory for all outputs
    config      : Fully-resolved PipelineCfg
    mode        : "auto" | "video" | "sequence" | "images"
    evaluate    : If True and a gt.txt is present, compute MOTA/MOTP
    """

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        config: PipelineCfg,
        mode: str = "auto",
        evaluate: bool = False,
        show_preview: bool = False,
    ) -> None:
        self._input = input_path
        self._output_dir = output_dir
        self._cfg = config
        self._mode = mode
        self._evaluate = evaluate
        self._show_preview = show_preview

        # Threading primitives
        self._stop_event = threading.Event()
        self._raw_q: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)
        self._det_q: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)

        # Lazy-initialised resources (created in run())
        self._source = None
        self._face_db: Optional[FaceDB] = None
        self._output_handler: Optional[OutputHandler] = None

        # Performance tracking
        self._frame_times: List[float] = []
        self._t_start: float = 0.0

    # ── public API ────────────────────────────────────────────────────── #

    def run(self) -> None:
        """Execute the full pipeline.  Blocks until completion."""
        self._t_start = time.perf_counter()

        # ── open source ── #
        self._source, eff_mode = open_source(self._input, self._cfg, self._mode)
        is_image_mode = (eff_mode == "images")
        logger.info("Effective mode: %s", eff_mode)

        # ── open database ── #
        self._face_db = FaceDB(self._cfg)

        # ── load zones ── #
        zones = load_zones(self._cfg.zones_path)

        # ── output handler ── #
        w, h = self._source.resolution
        fps  = self._source.fps
        self._output_handler = OutputHandler(
            output_dir=self._output_dir,
            cfg=self._cfg,
            source_fps=fps,
            source_resolution=(w, h),
        )

        # ── optional MOT evaluator ── #
        mot_evaluator: Optional[MOTEvaluator] = None
        if self._evaluate:
            gt_path = self._input / "gt" / "gt.txt"
            if gt_path.exists():
                mot_evaluator = MOTEvaluator(gt_path)
                logger.info("MOT evaluation enabled")
            else:
                logger.warning("--evaluate set but gt.txt not found at %s", gt_path)

        # ── start threads ── #
        reader_thread = ReaderThread(
            source=self._source,
            out_q=self._raw_q,
            stop_event=self._stop_event,
        )
        inference_thread = threading.Thread(
            target=self._inference_worker,
            name="InferenceThread",
            daemon=True,
        )
        analytics_thread = threading.Thread(
            target=self._analytics_output_worker,
            args=(zones, is_image_mode, mot_evaluator, self._show_preview),
            name="AnalyticsOutputThread",
            daemon=True,
        )

        reader_thread.start()
        inference_thread.start()
        analytics_thread.start()

        # ── wait for completion ── #
        analytics_thread.join()
        inference_thread.join()
        reader_thread.join()

        # ── final metrics ── #
        elapsed = time.perf_counter() - self._t_start
        if self._frame_times:
            avg_fps = len(self._frame_times) / elapsed
            logger.info(
                "Pipeline complete: %d frames in %.1f s  (%.1f fps avg)",
                len(self._frame_times), elapsed, avg_fps,
            )
        if mot_evaluator is not None:
            results = mot_evaluator.summary()
            logger.info("MOT results: %s", results)
            self._write_mot_results(results)

        db_count = self._face_db.total_identities() if self._face_db else 0
        logger.info("Total persistent identities in DB: %d", db_count)

    def cleanup(self) -> None:
        """Release all resources.  Safe to call multiple times."""
        self._stop_event.set()
        if self._source is not None:
            self._source.close()
        if self._output_handler is not None:
            self._output_handler.close()
            self._output_handler = None
        if self._face_db is not None:
            self._face_db.close()
            self._face_db = None

    # ── thread workers ─────────────────────────────────────────────────── #

    def _inference_worker(self) -> None:
        """
        Pulls RawFrames from raw_q, runs RT-DETR + FaceDetector,
        pushes InferenceResult to det_q.
        """
        try:
            detector = Detector(self._cfg)
            face_det = FaceDetector(self._cfg)
        except Exception as exc:
            logger.exception("InferenceThread: model init failed: %s", exc)
            self._det_q.put(_SENTINEL)
            return

        logger.info("InferenceThread started")

        while True:
            try:
                raw = self._raw_q.get(timeout=_QUEUE_TIMEOUT)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if raw is _SENTINEL:
                break

            try:
                persons, objects = detector.detect(raw.image)
                face_det.enrich_persons(persons, raw.image)
                result = InferenceResult(raw=raw, persons=persons, objects=objects)
            except Exception as exc:
                logger.warning("InferenceThread: frame %d failed: %s", raw.frame_idx, exc)
                result = InferenceResult(raw=raw, persons=[], objects=[])

            while True:
                try:
                    self._det_q.put(result, timeout=_QUEUE_TIMEOUT)
                    break
                except queue.Full:
                    if self._stop_event.is_set():
                        self._det_q.put(_SENTINEL)
                        return

            # Release image reference early to help GC
            del raw

        self._det_q.put(_SENTINEL)
        logger.info("InferenceThread finished")

    def _analytics_output_worker(
        self,
        zones: list,
        is_image_mode: bool,
        mot_evaluator: Optional[MOTEvaluator],
        show_preview: bool = False,
    ) -> None:
        """
        Pulls InferenceResults from det_q.
        Runs Tracking → ZoneLogic → CLIP → Visualizer → OutputHandler.
        """
        try:
            tracker_stage = None if is_image_mode else TrackerStage(self._cfg, self._face_db)
            zone_logic    = ZoneLogic(self._cfg, zones)
            anomaly_det   = AnomalyDetector(self._cfg)
            visualizer    = Visualizer(self._cfg, zones)
        except Exception as exc:
            logger.exception("AnalyticsOutputThread: init failed: %s", exc)
            return

        logger.info("AnalyticsOutputThread started (image_mode=%s)", is_image_mode)
        fps_smoother: List[float] = []
        t_prev = time.perf_counter()

        while True:
            try:
                inf = self._det_q.get(timeout=_QUEUE_TIMEOUT)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if inf is _SENTINEL:
                break

            t_now = time.perf_counter()
            frame_dt = t_now - t_prev
            t_prev = t_now
            fps_smoother.append(1.0 / max(frame_dt, 1e-6))
            if len(fps_smoother) > 30:
                fps_smoother.pop(0)
            current_fps = float(np.mean(fps_smoother))
            self._frame_times.append(t_now)

            try:
                analytics = self._process_analytics(
                    inf=inf,
                    tracker_stage=tracker_stage,
                    zone_logic=zone_logic,
                    anomaly_det=anomaly_det,
                    is_image_mode=is_image_mode,
                )

                # MOT evaluation update
                if mot_evaluator is not None and not is_image_mode:
                    hyp = self._tracks_to_mot_hyp(analytics.tracks)
                    mot_evaluator.update(inf.raw.frame_idx + 1, hyp)

                # Draw + write
                annotated = visualizer.draw(analytics, fps=current_fps)
                self._output_handler.write(analytics, annotated)

                # ── Live preview window (only when --preview flag is set) ── #
                if show_preview:
                    cv2.imshow("Surveillance Preview  [press Q to quit]", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # Q or Esc
                        logger.info("Preview window closed by user — stopping pipeline.")
                        self._stop_event.set()

                # ── Progress log every 100 frames so terminal stays alive ── #
                frame_idx = inf.raw.frame_idx
                if frame_idx % 100 == 0:
                    n_tracks = len(analytics.tracks)
                    logger.info(
                        "Frame %5d | fps=%.1f | tracks=%d",
                        frame_idx, current_fps, n_tracks,
                    )

            except Exception as exc:
                logger.warning(
                    "AnalyticsOutputThread: frame %d failed: %s",
                    inf.raw.frame_idx, exc,
                )

            del inf
            # Periodic GC to prevent fragmentation on long runs
            if len(self._frame_times) % 500 == 0:
                gc.collect()

        # Destroy the preview window cleanly when the loop exits
        if show_preview:
            cv2.destroyAllWindows()

        logger.info("AnalyticsOutputThread finished")

    # ── analytics pipeline (single frame) ─────────────────────────────── #

    def _process_analytics(
        self,
        inf: InferenceResult,
        tracker_stage: Optional[TrackerStage],
        zone_logic: ZoneLogic,
        anomaly_det: AnomalyDetector,
        is_image_mode: bool,
    ) -> AnalyticsResult:
        raw = inf.raw

        # ── Tracking (skipped in image mode) ── #
        if is_image_mode or tracker_stage is None:
            # In image mode, wrap detections directly as "tracks" with
            # face-DB-resolved permanent IDs but no session IDs
            tracks = self._detections_to_image_tracks(inf.persons, raw)
        else:
            tracked = tracker_stage.process(inf)
            tracks = tracked.tracks

        # ── Zone logic ── #
        zone_events, motion_activities, active_alerts = zone_logic.process(
            TrackedResult(raw=raw, tracks=tracks, objects=inf.objects)
        )

        # ── CLIP anomaly + per-person activity ── #
        scene_label, scene_conf, clip_activities = anomaly_det.analyse(
            frame_idx=raw.frame_idx,
            image=raw.image,
            tracks=tracks,
        )

        # Merge CLIP activities with motion-based ones
        # (CLIP overrides motion on CLIP-inference frames)
        person_activities: Dict[int, str] = {}
        for track in tracks:
            sid = track.session_id
            if sid in clip_activities:
                person_activities[sid] = clip_activities[sid]
            else:
                person_activities[sid] = motion_activities.get(sid, "Person is standing")

        # Per-person anomaly label (scene label applied to all for now)
        person_anomalies = {t.session_id: scene_label for t in tracks}

        # ── Build loggable events ── #
        loggable = self._build_loggable(
            raw, tracks, inf.objects, zone_events,
            person_activities, person_anomalies,
            scene_label, scene_conf,
        )

        return AnalyticsResult(
            raw=raw,
            tracks=tracks,
            objects=inf.objects,
            scene_anomaly=scene_label,
            scene_anomaly_confidence=scene_conf,
            person_activities=person_activities,
            person_anomalies=person_anomalies,
            zone_events=zone_events,
            loggable_events=loggable,
            active_alerts=active_alerts,
        )

    # ── helpers ────────────────────────────────────────────────────────── #

    def _detections_to_image_tracks(
        self,
        persons: List[Detection],
        raw: RawFrame,
    ) -> List[TrackInfo]:
        """
        For image-mode: resolve face-DB permanent IDs without ByteTrack.
        session_id is assigned sequentially per frame (no continuity).
        """
        tracks: List[TrackInfo] = []
        for idx, det in enumerate(persons):
            face_emb = det.face.embedding if det.face is not None else None
            pid, _ = self._face_db.find_or_create(
                face_emb=face_emb,
                body_emb=None,
                source=raw.source,
                timestamp=raw.timestamp,
            )
            tracks.append(TrackInfo(
                session_id=idx,
                permanent_id=pid,
                bbox=det.bbox.copy(),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name="person",
                is_face_identified=(face_emb is not None),
                face_embedding=face_emb,
            ))
        return tracks

    @staticmethod
    def _tracks_to_mot_hyp(tracks: List[TrackInfo]) -> np.ndarray:
        """Convert track list to (N, 5) array for MOT evaluator."""
        if not tracks:
            return np.empty((0, 5), dtype=np.float32)
        rows = []
        for t in tracks:
            rows.append([*t.bbox, float(t.session_id)])
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _build_loggable(
        raw, tracks, objects, zone_events,
        activities, person_anomalies,
        scene_label, scene_conf,
    ) -> List[Dict]:
        """Construct per-detection event dicts for the output handler."""
        events = []
        zone_by_sid = {e.session_id: e for e in zone_events}

        for t in tracks:
            ev = zone_by_sid.get(t.session_id)
            events.append({
                "frame_idx":           raw.frame_idx,
                "timestamp_s":         raw.timestamp,
                "permanent_id":        t.permanent_id,
                "session_id":          t.session_id,
                "object_class":        "person",
                "bbox":                t.bbox.tolist(),
                "confidence":          t.confidence,
                "activity":            activities.get(t.session_id, ""),
                "anomaly":             person_anomalies.get(t.session_id, "normal"),
                "scene_anomaly":       scene_label,
                "scene_anomaly_conf":  scene_conf,
                "zone_name":           ev.zone_name if ev else "",
                "event_type":          ev.event_type if ev else "PERSON_DETECTED",
            })
        for det in objects:
            events.append({
                "frame_idx":    raw.frame_idx,
                "timestamp_s":  raw.timestamp,
                "permanent_id": None,
                "session_id":   None,
                "object_class": det.class_name,
                "bbox":         det.bbox.tolist(),
                "confidence":   det.confidence,
                "event_type":   "OBJECT_DETECTED",
                "scene_anomaly": scene_label,
                "scene_anomaly_conf": scene_conf,
            })
        return events
    """

    def _write_mot_results(self, results: Dict) -> None:
        import json
        path = self._output_dir / "mot_evaluation.json"
        with path.open("w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("MOT evaluation results saved to %s", path)
    """

    def _write_mot_results(self, results: Dict) -> None:
        import json
        import numpy as np

        # Create a custom encoder to convert NumPy types to native Python types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        path = self._output_dir / "mot_evaluation.json"
        with path.open("w") as fh:
            # Pass our custom encoder into the dump function
            json.dump(results, fh, indent=2, cls=NumpyEncoder)
            
        logger.info("MOT evaluation results saved to %s", path)