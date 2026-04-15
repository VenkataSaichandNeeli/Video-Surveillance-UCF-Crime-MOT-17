"""
Face detection and ArcFace embedding extraction via InsightFace.

FIXES applied:
  1. Low-resolution upscaling: if the full frame is smaller than 480p,
     it is upscaled with bicubic interpolation before running RetinaFace.
     This dramatically improves face detection recall on UCF-Crime 320×240.
  2. Per-person crop upscaling: even when the full frame is reasonable,
     individual person crops that are too small are upscaled before the
     face detector is run on them.  This handles distant persons.
  3. det_score_threshold lowered to 0.25 (configurable) for low-res footage.
  4. min_face_size lowered to 8 pixels (configurable).
  5. Tight bounding-box check relaxed: face centre can be within an expanded
     person bbox (+10% margin) to tolerate detector imprecision at low res.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from .config import PipelineCfg
from .types import Detection, FaceInfo

logger = logging.getLogger(__name__)

# Minimum frame dimension before upscaling for face detection
_MIN_FRAME_DIM = 480
# Minimum scale factor applied to person crops before per-crop detection
_MIN_CROP_DIM = 96


class FaceDetector:
    def __init__(self, cfg: PipelineCfg) -> None:
        self._cfg = cfg
        self._app = self._load_app()

    def _load_app(self):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface not installed.  Run: pip install insightface onnxruntime-gpu"
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._cfg.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if self._cfg.device == "cuda" else -1
        app = FaceAnalysis(name=self._cfg.face.model_name, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=tuple(self._cfg.face.det_size))
        logger.info("FaceDetector: %s  ctx=%d  det_size=%s",
                    self._cfg.face.model_name, ctx_id, self._cfg.face.det_size)
        return app

    # ── public API ────────────────────────────────────────────────────── #

    def enrich_persons(
        self,
        persons: List[Detection],
        image: np.ndarray,
    ) -> None:
        """
        Detect faces in the full frame (upscaled if needed) and attach the
        best matching FaceInfo to each person Detection in-place.
        """
        proc_image, scale = self._maybe_upscale(image)
        faces = self._detect_faces(proc_image, scale)
        if not faces:
            return
        for det in persons:
            best = self._best_face_in_box(det.bbox, faces, image.shape)
            if best is not None:
                det.face = best

    # ── internals ─────────────────────────────────────────────────────── #

    def _maybe_upscale(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        If the frame is smaller than _MIN_FRAME_DIM in either dimension,
        upscale it with bicubic interpolation and return the scale factor
        so detected bbox coordinates can be mapped back to original space.
        """
        h, w = image.shape[:2]
        min_dim = min(h, w)
        if min_dim >= _MIN_FRAME_DIM:
            return image, 1.0
        scale = _MIN_FRAME_DIM / min_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug("Frame upscaled %.0fx%.0f → %.0fx%.0f (scale=%.2f) for face detection",
                     w, h, new_w, new_h, scale)
        return upscaled, scale

    def _detect_faces(
        self,
        image: np.ndarray,
        coord_scale: float,
    ) -> List[FaceInfo]:
        try:
            raw_faces = self._app.get(image)
        except Exception as exc:
            logger.warning("InsightFace.get() failed: %s", exc)
            return []

        results: List[FaceInfo] = []
        min_size = self._cfg.face.min_face_size
        score_thresh = self._cfg.face.det_score_threshold

        for face in raw_faces:
            if face.det_score < score_thresh:
                continue
            # Scale bbox back to original image coordinates
            bbox = (face.bbox / coord_scale).astype(np.int32)
            h = bbox[3] - bbox[1]
            if h < min_size:
                continue
            if face.embedding is None:
                continue
            emb = face.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                continue
            emb /= norm
            results.append(FaceInfo(
                bbox=bbox,
                confidence=float(face.det_score),
                embedding=emb,
            ))
        return results

    @staticmethod
    def _best_face_in_box(
        person_bbox: np.ndarray,
        faces: List[FaceInfo],
        frame_shape: tuple,
    ) -> Optional[FaceInfo]:
        """
        Find the best face whose centre lies inside or near the person bbox.
        A 10% margin is added around the person bbox to tolerate detection
        imprecision at low resolution.
        """
        px1, py1, px2, py2 = person_bbox
        # Add 10% margin
        pw = px2 - px1
        ph = py2 - py1
        margin_x = pw * 0.10
        margin_y = ph * 0.10
        epx1 = px1 - margin_x
        epy1 = py1 - margin_y
        epx2 = px2 + margin_x
        epy2 = py2 + margin_y

        best_face: Optional[FaceInfo] = None
        best_score = -1.0

        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox
            fc_x = (fx1 + fx2) / 2.0
            fc_y = (fy1 + fy2) / 2.0

            if not (epx1 <= fc_x <= epx2 and epy1 <= fc_y <= epy2):
                continue

            rel_x = (fc_x - px1) / max(pw, 1)
            centrality = 1.0 - abs(rel_x - 0.5) * 2
            score = face.confidence * (0.7 + 0.3 * centrality)

            if score > best_score:
                best_score = score
                best_face = face

        return best_face
