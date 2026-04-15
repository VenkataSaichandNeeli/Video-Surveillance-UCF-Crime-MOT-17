"""
Pipeline configuration: loads from JSON, applies CLI overrides, resolves device.

All numeric hyper-parameters live here; nothing is hard-coded elsewhere.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class DetectionCfg:
    model_name: str = "rtdetr-l.pt"
    confidence_threshold: float = 0.50
    person_class_id: int = 0
    imgsz: int = 640


@dataclass
class TrackingCfg:
    track_threshold: float = 0.50
    track_buffer: int = 30
    match_threshold: float = 0.80
    min_box_area: float = 100.0
    frame_rate: int = 30


@dataclass
class FaceCfg:
    model_name: str = "buffalo_l"
    det_size: Tuple[int, int] = (640, 640)
    det_score_threshold: float = 0.40
    # cosine-similarity threshold: same person if sim >= this value
    embedding_similarity_threshold: float = 0.35
    # seconds; lost track is eligible for re-ID within this window
    reidentification_window_seconds: float = 60.0
    # minimum face height in pixels; skip embedding if smaller
    min_face_size: int = 16


@dataclass
class ZoneCfg:
    loitering_threshold_seconds: float = 10.0
    loitering_movement_threshold_pixels: float = 50.0
    smoothing_window: int = 3          # frames to average bottom-centre point
    alert_cooldown_seconds: float = 30.0


@dataclass
class AnomalyCfg:
    model_name: str = "openai/clip-vit-base-patch32"
    inference_interval_frames: int = 15   # run CLIP every N frames
    scene_confidence_threshold: float = 0.30
    # activity thresholds (pixels/second, normalised to 1920-wide frame)
    activity_motion_standing_px_s: float = 10.0
    activity_motion_running_px_s: float = 80.0


@dataclass
class OutputCfg:
    save_annotated_video: bool = True
    save_event_frames: bool = True
    save_csv_log: bool = True
    log_all_detections: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.50
    alert_display_frames: int = 45      # how many frames an alert overlay persists


@dataclass
class PipelineCfg:
    detection: DetectionCfg = field(default_factory=DetectionCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    face: FaceCfg = field(default_factory=FaceCfg)
    zone: ZoneCfg = field(default_factory=ZoneCfg)
    anomaly: AnomalyCfg = field(default_factory=AnomalyCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    frame_sample_rate: int = 1       # process every Nth frame
    queue_maxsize: int = 32          # max frames in each inter-thread queue
    face_db_path: str = "data/face_identity.db"
    zones_path: str = "config/zones_template.json"
    device: str = "cpu"              # resolved at load time

    # ------------------------------------------------------------------ #

    @classmethod
    def load(
        cls,
        config_path: str,
        zones_path: str,
        device_override: str = "auto",
    ) -> PipelineCfg:
        """
        Build a config by:
          1. Starting from class defaults.
          2. Merging values from *config_path* (missing file is not an error).
          3. Applying the device override.
        """
        cfg = cls()
        cfg.zones_path = zones_path

        # ── Resolve compute device ── #
        if device_override == "auto":
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
            if cfg.device == "cuda":
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info("GPU found: %s  (%.1f GB VRAM)", name, vram)
            else:
                logger.warning(
                    "No CUDA GPU detected — running on CPU.  "
                    "Expect significantly lower throughput."
                )
        else:
            cfg.device = device_override

        # ── Merge JSON file ── #
        p = Path(config_path)
        if p.exists():
            try:
                with p.open() as fh:
                    data: Dict[str, Any] = json.load(fh)
                _merge(cfg, data)
                logger.info("Config loaded from %s", config_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not parse %s (%s) — using defaults.", config_path, exc
                )
        else:
            logger.info("Config file %s not found — using defaults.", config_path)

        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── helpers ────────────────────────────────────────────────────────────── #

def _merge(cfg: PipelineCfg, data: Dict[str, Any]) -> None:
    """Overwrite cfg fields from *data* without raising on unknown keys."""
    sub_map = {
        "detection": cfg.detection,
        "tracking": cfg.tracking,
        "face": cfg.face,
        "zone": cfg.zone,
        "anomaly": cfg.anomaly,
        "output": cfg.output,
    }
    for section, obj in sub_map.items():
        if section in data and isinstance(data[section], dict):
            for k, v in data[section].items():
                if hasattr(obj, k):
                    # det_size stored as list in JSON → convert to tuple
                    if k == "det_size" and isinstance(v, list):
                        v = tuple(v)
                    setattr(obj, k, v)

    for top_key in ("frame_sample_rate", "queue_maxsize", "face_db_path"):
        if top_key in data:
            setattr(cfg, top_key, data[top_key])