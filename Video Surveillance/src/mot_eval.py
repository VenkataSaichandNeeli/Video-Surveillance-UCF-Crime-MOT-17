"""
MOT evaluation: compute MOTA and MOTP against MOT17 gt.txt ground truth.

MOTA = 1 - (FP + FN + IDSW) / GT
MOTP = sum(IoU of matched pairs) / number of matches

Usage
-----
    evaluator = MOTEvaluator(gt_path)
    for each frame:
        evaluator.update(frame_id, hypothesis_boxes)
    results = evaluator.summary()
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_IOU_THRESHOLD = 0.50   # standard MOT matching threshold


class MOTEvaluator:
    """Accumulates per-frame metrics and produces a summary."""

    def __init__(self, gt_path: Path) -> None:
        self._gt = self._load_gt(gt_path)
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._idsw = 0
        self._iou_sum = 0.0
        self._match_count = 0
        # For IDSW: gt_id → last matched hyp_id
        self._gt_last_hyp: Dict[int, Optional[int]] = {}

    # ── public API ────────────────────────────────────────────────────── #

    def update(
        self,
        frame_id: int,          # 1-based, matching MOT17 convention
        hyp_boxes: np.ndarray,  # (N, 5): [x1, y1, x2, y2, track_id]
    ) -> None:
        """Process one frame."""
        gt_boxes = self._gt.get(frame_id, np.empty((0, 6)))
        # Filter: only class 1 (pedestrian), not-ignored (col 6 == 1)
        if len(gt_boxes) > 0:
            gt_boxes = gt_boxes[gt_boxes[:, 6] == 1]

        if len(gt_boxes) == 0 and len(hyp_boxes) == 0:
            return

        if len(gt_boxes) == 0:
            self._fp += len(hyp_boxes)
            return

        if len(hyp_boxes) == 0:
            self._fn += len(gt_boxes)
            return

        # Convert MOT xywh → xyxy for GT
        gt_xyxy = self._mot_to_xyxy(gt_boxes[:, 2:6])
        hyp_xyxy = hyp_boxes[:, :4]

        # Greedy IoU matching
        iou_mat = self._iou_matrix(gt_xyxy, hyp_xyxy)
        matched_gt, matched_hyp = self._greedy_match(iou_mat, _IOU_THRESHOLD)

        self._tp   += len(matched_gt)
        self._fp   += len(hyp_boxes) - len(matched_hyp)
        self._fn   += len(gt_boxes)  - len(matched_gt)
        self._iou_sum    += sum(iou_mat[g, h] for g, h in zip(matched_gt, matched_hyp))
        self._match_count += len(matched_gt)

        # IDSW: GT matched to a different hyp than last time
        gt_ids  = gt_boxes[:, 1].astype(int)
        hyp_ids = hyp_boxes[:, 4].astype(int) if hyp_boxes.shape[1] > 4 else np.zeros(len(hyp_boxes), int)
        for gi, hi in zip(matched_gt, matched_hyp):
            gt_id  = int(gt_ids[gi])
            hyp_id = int(hyp_ids[hi])
            last   = self._gt_last_hyp.get(gt_id)
            if last is not None and last != hyp_id:
                self._idsw += 1
            self._gt_last_hyp[gt_id] = hyp_id

    def summary(self) -> Dict[str, float]:
        """Return MOTA, MOTP, and component counts."""
        gt_total = self._tp + self._fn
        if gt_total == 0:
            mota = 0.0
        else:
            mota = 1.0 - (self._fp + self._fn + self._idsw) / gt_total

        motp = (
            self._iou_sum / self._match_count
            if self._match_count > 0
            else 0.0
        )

        report = {
            "MOTA":          round(mota, 4),
            "MOTP":          round(motp, 4),
            "TP":            self._tp,
            "FP":            self._fp,
            "FN":            self._fn,
            "IDSW":          self._idsw,
            "GT_total":      self._tp + self._fn,
            "Matches":       self._match_count,
        }
        logger.info("MOT evaluation: %s", report)
        return report

    # ── internals ─────────────────────────────────────────────────────── #

    @staticmethod
    def _load_gt(path: Path) -> Dict[int, np.ndarray]:
        """
        gt.txt columns (MOT format):
          frame, track_id, x, y, w, h, not_ignored, class_id, visibility
        """
        rows: Dict[int, list] = defaultdict(list)
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                try:
                    fid = int(parts[0])
                    row = [float(p) for p in parts[:9]]
                    rows[fid].append(row)
                except ValueError:
                    continue
        return {fid: np.array(v, dtype=np.float32) for fid, v in rows.items()}

    @staticmethod
    def _mot_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        """Convert MOT [x, y, w, h] top-left format to [x1,y1,x2,y2]."""
        out = xywh.copy()
        out[:, 2] = xywh[:, 0] + xywh[:, 2]
        out[:, 3] = xywh[:, 1] + xywh[:, 3]
        return out

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute (|a|, |b|) IoU matrix."""
        ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

        inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
        inter_y1 = np.maximum(ay1[:, None], by1[None, :])
        inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
        inter_y2 = np.minimum(ay2[:, None], by2[None, :])

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter   = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union  = area_a[:, None] + area_b[None, :] - inter

        return np.where(union > 0, inter / union, 0.0)

    @staticmethod
    def _greedy_match(
        iou_mat: np.ndarray,
        threshold: float,
    ) -> Tuple[List[int], List[int]]:
        """Greedy 1-to-1 matching: highest IoU first."""
        matched_g: List[int] = []
        matched_h: List[int] = []
        used_g: set = set()
        used_h: set = set()

        flat_indices = np.argsort(iou_mat.ravel())[::-1]
        ng, nh = iou_mat.shape
        for idx in flat_indices:
            gi, hi = divmod(int(idx), nh)
            if iou_mat[gi, hi] < threshold:
                break
            if gi in used_g or hi in used_h:
                continue
            matched_g.append(gi)
            matched_h.append(hi)
            used_g.add(gi)
            used_h.add(hi)

        return matched_g, matched_h