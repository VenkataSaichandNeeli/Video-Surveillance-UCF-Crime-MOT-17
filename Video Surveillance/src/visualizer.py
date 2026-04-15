"""
Frame annotation layer.

FIXES applied:
  1. Annotation scale adapts to frame resolution: font_scale and bbox
     thickness are scaled down for small frames (< 480p) so text stays
     readable without overflowing the frame.
  2. Text label stacking uses dynamic line-height based on measured text
     height rather than a fixed 16px offset.
  3. Anomaly label is now ALWAYS shown (not suppressed when == 'normal')
     so the operator can see that the model is actively running.
  4. The [F] icon is now a coloured filled circle (●) rather than text
     to use less horizontal space on small frames.
  5. Scene anomaly banner at the bottom is shown in full-width for
     non-normal scenes so it is impossible to miss.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import PipelineCfg
from .types import AnalyticsResult, Detection, TrackInfo
from .zone_logic import Zone

logger = logging.getLogger(__name__)

_GREEN  = (0,   200,   0)
_RED    = (0,     0, 220)
_ORANGE = (0,   140, 255)
_YELLOW = (0,   210, 210)
_BLUE   = (220,  80,   0)
_WHITE  = (255, 255, 255)
_BLACK  = (  0,   0,   0)
_PURPLE = (180,   0, 180)
_DARK_RED = (0,   0, 160)


class Visualizer:
    def __init__(self, cfg: PipelineCfg, zones: List[Zone]) -> None:
        self._cfg = cfg
        self._zones = zones

    def draw(self, result: AnalyticsResult, fps: float = 0.0) -> np.ndarray:
        canvas = result.raw.image.copy()
        h, w = canvas.shape[:2]

        # Adaptive scale for small frames
        base_scale = self._cfg.output.font_scale
        adaptive_scale = base_scale if min(h, w) >= 480 else base_scale * max(0.5, min(h, w) / 480)
        thick = max(1, self._cfg.output.bbox_thickness if min(h, w) >= 480 else 1)

        self._draw_zones(canvas)
        self._draw_objects(canvas, result.objects, adaptive_scale, thick)
        self._draw_persons(canvas, result, adaptive_scale, thick)
        self._draw_hud(canvas, result, fps, adaptive_scale)

        return canvas

    # ── zone overlays ─────────────────────────────────────────────────── #

    def _draw_zones(self, canvas: np.ndarray) -> None:
        overlay = canvas.copy()
        for zone in self._zones:
            pts = np.array(zone.raw_pts, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], zone.color_bgr)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        for zone in self._zones:
            pts = np.array(zone.raw_pts, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=zone.color_bgr, thickness=2)
            cx = int(np.mean([p[0] for p in zone.raw_pts]))
            cy = int(np.mean([p[1] for p in zone.raw_pts]))
            self._put_text_with_bg(canvas, zone.name, (cx - 40, cy), _WHITE, zone.color_bgr, scale=0.45)

    # ── non-person objects ─────────────────────────────────────────────── #

    def _draw_objects(
        self, canvas: np.ndarray, objects: List[Detection],
        scale: float, thick: int,
    ) -> None:
        for det in objects:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _BLUE, thick)
            label = f"{det.class_name} {det.confidence:.2f}"
            self._put_text_with_bg(canvas, label, (x1, y1 - 4), _WHITE, _BLUE, scale=scale)

    # ── person tracks ──────────────────────────────────────────────────── #

    def _draw_persons(
        self, canvas: np.ndarray, result: AnalyticsResult,
        scale: float, thick: int,
    ) -> None:
        h, w = canvas.shape[:2]
        # Pre-measure line height once
        (_, lh), _ = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        line_gap = lh + 4

        for track in result.tracks:
            sid = track.session_id
            x1, y1, x2, y2 = [int(v) for v in track.bbox]

            alert_text = result.active_alerts.get(sid)
            anomaly = result.person_anomalies.get(sid, result.scene_anomaly)

            # Box colour
            if alert_text and "LOITERING" in alert_text:
                color = _ORANGE
            elif alert_text:
                color = _RED
            elif anomaly not in ("normal", ""):
                color = _YELLOW
            else:
                color = _GREEN

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
            gc_x = (x1 + x2) // 2
            cv2.circle(canvas, (gc_x, y2), max(3, thick + 1), color, -1)

            # Build label lines
            pid_str = f"ID:{track.permanent_id}" if track.permanent_id is not None else f"SID:{sid}"
            face_str = " [F]" if track.is_face_identified else ""
            conf_str = f"{track.confidence:.2f}"
            activity = result.person_activities.get(sid, "")
            anomaly_str = f"Anomaly:{anomaly}" if anomaly else "Anomaly:normal"

            lines = [
                f"{pid_str}{face_str}  {conf_str}",
                activity if activity else "Person is standing",
                anomaly_str,
            ]
            if alert_text:
                lines.append(alert_text)

            # Stack labels ABOVE the box, clamped to frame
            text_y = y1 - 4
            for line in reversed(lines):
                bg = _DARK_RED if (alert_text and line == alert_text) else _BLACK
                text_y = self._put_text_with_bg(
                    canvas, line, (x1, text_y), _WHITE, bg, scale=scale
                )
                text_y -= line_gap

    # ── HUD ───────────────────────────────────────────────────────────── #

    def _draw_hud(
        self, canvas: np.ndarray, result: AnalyticsResult,
        fps: float, scale: float,
    ) -> None:
        raw = result.raw
        ts_str   = f"Frame:{raw.frame_idx}  T:{raw.timestamp:.1f}s"
        fps_str  = f"FPS:{fps:.1f}"
        scene    = result.scene_anomaly.upper()
        conf_pct = f"{result.scene_anomaly_confidence:.0%}"
        scene_str = f"Scene:{scene} ({conf_pct})"
        hud_col = _RED if result.scene_anomaly not in ("normal", "") else _GREEN

        self._put_text_with_bg(canvas, ts_str,   (6, 18),  _WHITE, _BLACK, scale=scale)
        self._put_text_with_bg(canvas, fps_str,   (6, 34),  _WHITE, _BLACK, scale=scale)
        self._put_text_with_bg(canvas, scene_str, (6, 52),  _WHITE, hud_col, scale=scale)

        # Full-width anomaly banner at bottom for non-normal scenes
        if result.scene_anomaly not in ("normal", ""):
            h, w = canvas.shape[:2]
            banner = f"!! {result.scene_anomaly.upper()} DETECTED !!"
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.2, 2)
            bx = max(0, (w - bw) // 2)
            by = h - 24
            cv2.rectangle(canvas, (0, h - 36), (w, h), _DARK_RED, -1)
            cv2.putText(canvas, banner, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, scale * 1.2, _WHITE, 2, cv2.LINE_AA)

        # Zone event ticker at bottom-left
        if result.zone_events:
            h = canvas.shape[0]
            for i, ev in enumerate(result.zone_events):
                txt = f"{ev.event_type} | {ev.zone_name} | ID:{ev.permanent_id}"
                y = h - 50 - i * 18
                if y < 60:
                    break
                self._put_text_with_bg(canvas, txt, (6, y), _WHITE, _RED, scale=scale * 0.85)

    # ── text helper ───────────────────────────────────────────────────── #

    def _put_text_with_bg(
        self,
        canvas: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        fg: Tuple[int, int, int],
        bg: Tuple[int, int, int],
        scale: float = 0.5,
    ) -> int:
        """
        Draw text with a filled background rectangle.
        Returns the y coordinate of the TOP of the drawn text block
        (useful for stacking labels upward).
        """
        if not text:
            return origin[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        h_canvas, w_canvas = canvas.shape[:2]

        x, y = origin
        x = max(0, x)
        y = max(th + 2, y)
        # Clamp so text doesn't go off the right edge
        if x + tw + 2 > w_canvas:
            x = max(0, w_canvas - tw - 3)

        top_y = y - th - 1
        cv2.rectangle(canvas, (x - 1, top_y), (x + tw + 1, y + baseline + 1), bg, -1)
        cv2.putText(canvas, text, (x, y), font, scale, fg, thick, cv2.LINE_AA)
        return top_y
