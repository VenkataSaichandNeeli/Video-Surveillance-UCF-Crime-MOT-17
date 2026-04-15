"""
Object detection via RT-DETRv2 (ultralytics).

RT-DETR is NMS-free, which means it does not suppress overlapping detections
of different people in crowded scenes — a deliberate advantage over YOLO.

The inference thread calls `Detector.detect()` once per frame and receives
a flat list of Detection objects separated into persons vs. other objects.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import PipelineCfg
from .types import Detection

logger = logging.getLogger(__name__)

# COCO class-id 0 → "person"
_PERSON_CLASS = 0

# COCO label map (subset; extend if needed)
_COCO_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 13: "bench", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 62: "tv",
    63: "laptop", 67: "cell phone", 73: "book", 76: "scissors",
}


class Detector:
    """
    Wraps ultralytics RT-DETR for single-frame inference.

    Thread-safety: NOT thread-safe — create one instance per thread.
    """

    def __init__(self, cfg: PipelineCfg) -> None:
        self._cfg = cfg
        self._model = self._load_model()

    # ── construction ──────────────────────────────────────────────────── #

    def _load_model(self):
        try:
            from ultralytics import RTDETR  # type: ignore[import]
            model = RTDETR(self._cfg.detection.model_name)
            model.to(self._cfg.device)
            model.fuse()    # fuse conv+bn layers for speed
            logger.info(
                "RT-DETR loaded: %s on %s",
                self._cfg.detection.model_name, self._cfg.device,
            )
            return model
        except Exception as exc:
            logger.error("Failed to load RT-DETR: %s", exc)
            raise

    # ── public API ────────────────────────────────────────────────────── #

    def detect(
        self,
        image: np.ndarray,
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Run detection on one BGR frame.

        Returns
        -------
        persons : List[Detection]
            All person detections above conf threshold.
        objects : List[Detection]
            All non-person detections above conf threshold.
        """
        conf_thresh = self._cfg.detection.confidence_threshold

        with torch.no_grad():
            results = self._model.predict(
                image,
                imgsz=self._cfg.detection.imgsz,
                conf=conf_thresh,
                verbose=False,
                device=self._cfg.device,
            )

        persons: List[Detection] = []
        objects: List[Detection] = []

        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()    # (N, 4)
            confs = r.boxes.conf.cpu().numpy()    # (N,)
            classes = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for i in range(len(xyxy)):
                cls_id = int(classes[i])
                conf = float(confs[i])
                bbox = xyxy[i].astype(np.float32)
                name = _COCO_NAMES.get(cls_id, f"class_{cls_id}")
                det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=name,
                )
                if cls_id == _PERSON_CLASS:
                    persons.append(det)
                else:
                    objects.append(det)

        return persons, objects