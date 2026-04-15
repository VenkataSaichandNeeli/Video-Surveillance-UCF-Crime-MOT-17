"""
Interactive polygon zone editor using OpenCV's HighGUI.

Usage
-----
    editor = ZoneEditor(input_path, zones_path, cfg)
    editor.run()   # blocks until user saves or quits

Controls
--------
  Left-click       Add a polygon vertex
  Right-click      Close current polygon (needs ≥ 3 points)
  Z                Undo last vertex
  N                Rename the zone being drawn (prompts on terminal)
  T                Toggle zone type (restricted / monitored / entry_exit)
  C                Clear current in-progress polygon
  S                Save all zones and exit
  Q / Esc          Quit without saving

A frame is extracted from the input source to serve as the backdrop.
Completed zones are saved to *zones_path* as the standard JSON format.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineCfg

logger = logging.getLogger(__name__)

_ZONE_COLORS = [
    (0,   0,   220),    # red
    (0,   180,  0),     # green
    (200,  80,   0),    # blue
    (0,   180, 255),    # orange
    (180,   0, 180),    # purple
    (0,   220, 220),    # yellow
]

_TYPE_CYCLE = ["restricted", "monitored", "entry_exit"]


class ZoneEditor:
    """
    Blocking interactive editor.  Call `run()` from the main thread.
    """

    def __init__(
        self,
        input_path: Path,
        zones_path: Path,
        cfg: PipelineCfg,
    ) -> None:
        self._input = input_path
        self._zones_path = zones_path
        self._cfg = cfg

        self._saved_zones: List[dict] = []        # completed zones
        self._current_pts: List[Tuple[int, int]] = []
        self._current_name = "Zone 1"
        self._current_type_idx = 0                # index into _TYPE_CYCLE

        self._bg_frame: Optional[np.ndarray] = None
        self._display:  Optional[np.ndarray] = None
        self._mouse_pos: Tuple[int, int] = (0, 0)

    # ── public ────────────────────────────────────────────────────────── #

    def run(self) -> None:
        self._bg_frame = self._extract_frame()
        if self._bg_frame is None:
            logger.error("ZoneEditor: could not extract a frame from %s", self._input)
            return

        # Load existing zones so user can see / extend them
        if self._zones_path.exists():
            try:
                with self._zones_path.open() as fh:
                    data = json.load(fh)
                self._saved_zones = data.get("zones", [])
                logger.info("Loaded %d existing zones", len(self._saved_zones))
            except Exception:
                pass

        self._redraw()
        cv2.namedWindow("Zone Editor — press H for help", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "Zone Editor — press H for help",
            min(1280, self._bg_frame.shape[1]),
            min(720,  self._bg_frame.shape[0]),
        )
        cv2.setMouseCallback("Zone Editor — press H for help", self._on_mouse)

        print(
            "\n── Zone Editor ────────────────────────────────────────\n"
            "  Left-click  : add vertex\n"
            "  Right-click : close polygon (≥ 3 points required)\n"
            "  Z           : undo last vertex\n"
            "  N           : rename current zone\n"
            "  T           : toggle zone type\n"
            "  C           : clear current polygon\n"
            "  S           : save and exit\n"
            "  Q / Esc     : quit without saving\n"
            "────────────────────────────────────────────────────────\n"
        )

        while True:
            cv2.imshow("Zone Editor — press H for help", self._display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self._save()
                break
            elif key == ord('z'):
                if self._current_pts:
                    self._current_pts.pop()
                    self._redraw()
            elif key == ord('c'):
                self._current_pts.clear()
                self._redraw()
            elif key == ord('n'):
                name = input(f"Zone name [{self._current_name}]: ").strip()
                if name:
                    self._current_name = name
                self._redraw()
            elif key == ord('t'):
                self._current_type_idx = (self._current_type_idx + 1) % len(_TYPE_CYCLE)
                print(f"Zone type → {_TYPE_CYCLE[self._current_type_idx]}")
                self._redraw()

        cv2.destroyAllWindows()

    # ── mouse callback ─────────────────────────────────────────────────── #

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        self._mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_pts.append((x, y))
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self._current_pts) >= 3:
                self._close_polygon()
            else:
                print("Need at least 3 points to close a polygon.")

        elif event == cv2.EVENT_MOUSEMOVE:
            self._redraw()   # live rubber-band line

    # ── drawing ────────────────────────────────────────────────────────── #

    def _redraw(self) -> None:
        canvas = self._bg_frame.copy()

        # ── completed zones ── #
        for i, zone in enumerate(self._saved_zones):
            pts = np.array(zone["polygon_2d"], dtype=np.int32)
            color = _ZONE_COLORS[i % len(_ZONE_COLORS)]
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
            cv2.polylines(canvas, [pts], True, color, 2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            _put_label(canvas, f"{zone['name']} [{zone['type']}]", (cx - 50, cy), color)

        # ── in-progress polygon ── #
        if self._current_pts:
            pts_arr = np.array(self._current_pts, dtype=np.int32)
            in_prog_color = _ZONE_COLORS[len(self._saved_zones) % len(_ZONE_COLORS)]
            for pt in pts_arr:
                cv2.circle(canvas, tuple(pt), 5, in_prog_color, -1)
            if len(pts_arr) >= 2:
                cv2.polylines(canvas, [pts_arr], False, in_prog_color, 1)
            # rubber-band to mouse
            cv2.line(canvas, self._current_pts[-1], self._mouse_pos, in_prog_color, 1)

        # ── status bar ── #
        zone_type = _TYPE_CYCLE[self._current_type_idx]
        status = (
            f"Zone: '{self._current_name}'  Type: {zone_type}  "
            f"Vertices: {len(self._current_pts)}  "
            f"Saved zones: {len(self._saved_zones)}"
        )
        _put_label(canvas, status, (8, canvas.shape[0] - 10), (255, 255, 0))
        self._display = canvas

    def _close_polygon(self) -> None:
        color_hex = "#{:02x}{:02x}{:02x}".format(
            *_ZONE_COLORS[len(self._saved_zones) % len(_ZONE_COLORS)][::-1]
        )
        zone = {
            "name":                       self._current_name,
            "type":                       _TYPE_CYCLE[self._current_type_idx],
            "polygon_2d":                 [list(p) for p in self._current_pts],
            "polygon_3d":                 [[p[0], p[1], 0] for p in self._current_pts],
            "loitering_threshold_seconds": self._cfg.zone.loitering_threshold_seconds,
            "color":                      color_hex,
        }
        self._saved_zones.append(zone)
        print(f"Zone '{self._current_name}' saved ({len(self._current_pts)} vertices).")
        self._current_pts = []
        self._current_name = f"Zone {len(self._saved_zones) + 1}"
        self._current_type_idx = 0
        self._redraw()

    # ── I/O ──────────────────────────────────────────────────────────── #

    def _save(self) -> None:
        self._zones_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"zones": self._saved_zones}
        with self._zones_path.open("w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nSaved {len(self._saved_zones)} zone(s) to {self._zones_path}\n")

    def _extract_frame(self) -> Optional[np.ndarray]:
        """Pull the first readable frame from any supported input type."""
        p = self._input.resolve()

        if p.is_file():
            cap = cv2.VideoCapture(str(p))
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame

        if (p / "seqinfo.ini").exists():
            img_dir = p / "img1"
        elif p.is_dir():
            img_dir = p
        else:
            return None

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    return frame

        return None


# ── helper ────────────────────────────────────────────────────────────── #

def _put_label(
    canvas: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
    x, y = origin
    cv2.rectangle(canvas, (x - 1, y - th - 2), (x + tw + 1, y + 2), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), font, 0.5, color, 1, cv2.LINE_AA)