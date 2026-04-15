"""
Output handler: event CSV logging + annotated video writing.

FIX (video-in-corner bug): The VideoWriter is NOT opened at construction time.
It is opened lazily on the first call to write(), using the ACTUAL dimensions
of the annotated frame.  This prevents the mismatch between
CAP_PROP_FRAME_WIDTH/HEIGHT (which can be wrong for rotated UCF-Crime clips)
and the real decoded frame size, which caused content to appear only in a
corner of a larger black canvas.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import PipelineCfg
from .types import AnalyticsResult

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "wall_time", "frame_idx", "timestamp_s", "source",
    "permanent_id", "session_id", "object_class",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "confidence", "face_identified",
    "activity", "anomaly", "scene_anomaly", "scene_anomaly_conf",
    "zone_name", "event_type",
]


class OutputHandler:
    def __init__(
        self,
        output_dir: Path,
        cfg: PipelineCfg,
        source_fps: float,
        source_resolution: tuple,
    ) -> None:
        self._cfg = cfg
        self._out_dir = output_dir
        self._fps = max(source_fps, 1.0)
        self._hint_w, self._hint_h = source_resolution

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample_frames").mkdir(exist_ok=True)

        # VideoWriter opened lazily on first frame — uses REAL decoded size
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_path = output_dir / "output.mp4"
        self._actual_w: int = 0
        self._actual_h: int = 0

        self._csv_file = None
        self._csv_writer = None

        if cfg.output.save_csv_log:
            csv_path = output_dir / "events.csv"
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=_CSV_COLUMNS, extrasaction="ignore"
            )
            self._csv_writer.writeheader()
            self._csv_file.flush()
            logger.info("Event log: %s", csv_path)

        self._frame_count = 0
        self._event_count = 0

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
        logger.info("OutputHandler closed: %d frames, %d events", self._frame_count, self._event_count)

    def __enter__(self):
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def write(self, result: AnalyticsResult, annotated_frame: np.ndarray) -> None:
        if annotated_frame is None:
            return

        if self._cfg.output.save_annotated_video:
            fh, fw = annotated_frame.shape[:2]
            if self._video_writer is None:
                # Use ACTUAL frame size, not metadata size
                self._actual_w = fw
                self._actual_h = fh
                self._video_writer = self._open_video(self._video_path, fw, fh)
                if (fw, fh) != (self._hint_w, self._hint_h):
                    logger.warning(
                        "Frame size mismatch: metadata=%dx%d actual=%dx%d — "
                        "VideoWriter uses actual size (this was the video-in-corner bug).",
                        self._hint_w, self._hint_h, fw, fh,
                    )
            # Guard: resize if somehow a frame differs
            if fw != self._actual_w or fh != self._actual_h:
                annotated_frame = cv2.resize(
                    annotated_frame, (self._actual_w, self._actual_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            self._video_writer.write(annotated_frame)

        if self._csv_writer is not None and self._cfg.output.log_all_detections:
            self._log_all_detections(result)

        if self._cfg.output.save_event_frames and result.zone_events:
            self._save_event_frame(result, annotated_frame)

        self._frame_count += 1

    def _log_all_detections(self, result: AnalyticsResult) -> None:
        raw = result.raw
        wall = time.time()

        for track in result.tracks:
            sid = track.session_id
            zone_ev = next((e for e in result.zone_events if e.session_id == sid), None)
            row: Dict[str, Any] = {
                "wall_time": wall, "frame_idx": raw.frame_idx,
                "timestamp_s": round(raw.timestamp, 3), "source": raw.source,
                "permanent_id": track.permanent_id, "session_id": sid,
                "object_class": "person",
                "bbox_x1": round(float(track.bbox[0]), 1),
                "bbox_y1": round(float(track.bbox[1]), 1),
                "bbox_x2": round(float(track.bbox[2]), 1),
                "bbox_y2": round(float(track.bbox[3]), 1),
                "confidence": round(track.confidence, 3),
                "face_identified": track.is_face_identified,
                "activity": result.person_activities.get(sid, ""),
                "anomaly": result.person_anomalies.get(sid, "normal"),
                "scene_anomaly": result.scene_anomaly,
                "scene_anomaly_conf": round(result.scene_anomaly_confidence, 3),
                "zone_name": zone_ev.zone_name if zone_ev else "",
                "event_type": zone_ev.event_type if zone_ev else "PERSON_DETECTED",
            }
            self._csv_writer.writerow(row)
            self._event_count += 1

        for det in result.objects:
            row = {
                "wall_time": wall, "frame_idx": raw.frame_idx,
                "timestamp_s": round(raw.timestamp, 3), "source": raw.source,
                "permanent_id": None, "session_id": None,
                "object_class": det.class_name,
                "bbox_x1": round(float(det.bbox[0]), 1),
                "bbox_y1": round(float(det.bbox[1]), 1),
                "bbox_x2": round(float(det.bbox[2]), 1),
                "bbox_y2": round(float(det.bbox[3]), 1),
                "confidence": round(det.confidence, 3),
                "face_identified": False, "activity": "", "anomaly": "",
                "scene_anomaly": result.scene_anomaly,
                "scene_anomaly_conf": round(result.scene_anomaly_confidence, 3),
                "zone_name": "", "event_type": "OBJECT_DETECTED",
            }
            self._csv_writer.writerow(row)
            self._event_count += 1

        if self._frame_count % 100 == 0:
            self._csv_file.flush()

    def _save_event_frame(self, result: AnalyticsResult, annotated: np.ndarray) -> None:
        seen: set = set()
        for ev in result.zone_events:
            key = f"{ev.event_type}_{ev.zone_name}_{result.raw.frame_idx}"
            if key in seen:
                continue
            seen.add(key)
            fname = (
                f"frame{result.raw.frame_idx:06d}"
                f"_{ev.event_type}_{ev.zone_name.replace(' ', '_')}.jpg"
            )
            cv2.imwrite(str(self._out_dir / "sample_frames" / fname), annotated)

    def _open_video(self, path: Path, w: int, h: int) -> cv2.VideoWriter:
        """Try codecs in priority order for maximum player compatibility."""
        for codec in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))
            if writer.isOpened():
                logger.info("VideoWriter: %s  codec=%s  %dx%d @ %.1f fps",
                            path.name, codec, w, h, self._fps)
                return writer
            writer.release()
        raise IOError(f"Cannot open VideoWriter at {path}")
