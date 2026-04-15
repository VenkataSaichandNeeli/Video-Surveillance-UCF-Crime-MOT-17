"""
Ingestion layer: unified iterator over frames from three source types.

  VideoIngestion    – any OpenCV-readable video file
  MOT17Ingestion    – image sequence with seqinfo.ini + optional det.txt
  ImageIngestion    – flat folder of images (no temporal continuity)

The public factory `open_source()` inspects the path and returns the
right implementation.  Wrap the result in a `ReaderThread` to push frames
into the pipeline queue in a background thread.
"""
from __future__ import annotations

import configparser
import logging
import os
import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .config import PipelineCfg
from .types import RawFrame

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ══════════════════════════════════════════════════════════════════════════ #
#  Abstract base
# ══════════════════════════════════════════════════════════════════════════ #

class BaseIngestion(ABC):
    """Yields RawFrame objects; must be used as a context manager."""

    @abstractmethod
    def __iter__(self) -> Iterator[RawFrame]:
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        ...

    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Best-effort estimate; -1 if unknown."""
        ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(width, height)"""
        ...

    def close(self) -> None:  # noqa: B027 – subclasses may override
        pass

    def __enter__(self) -> BaseIngestion:
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ══════════════════════════════════════════════════════════════════════════ #
#  Video file
# ══════════════════════════════════════════════════════════════════════════ #

class VideoIngestion(BaseIngestion):
    """Reads any video file OpenCV can decode."""

    def __init__(self, path: Path, cfg: PipelineCfg) -> None:
        self._path = path
        self._cfg = cfg
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "VideoIngestion: %s  (%dx%d @ %.1f fps, ~%d frames)",
            path.name, self._w, self._h, self._fps, self._total,
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def resolution(self) -> tuple[int, int]:
        return self._w, self._h

    def __iter__(self) -> Iterator[RawFrame]:
        sample = self._cfg.frame_sample_rate
        global_idx = 0
        read_idx = 0
        source = str(self._path)

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            read_idx += 1
            if (read_idx - 1) % sample != 0:
                continue
            timestamp = (read_idx - 1) / self._fps
            yield RawFrame(
                frame_idx=global_idx,
                timestamp=timestamp,
                image=frame,
                source=source,
                fps=self._fps,
            )
            global_idx += 1

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()


# ══════════════════════════════════════════════════════════════════════════ #
#  MOT17 image sequence
# ══════════════════════════════════════════════════════════════════════════ #

class MOT17Ingestion(BaseIngestion):
    """
    Reads a MOT17-style sequence directory:

        seq_dir/
            seqinfo.ini
            img1/  001.jpg  002.jpg …
            det/   det.txt          (optional, loaded if present)
            gt/    gt.txt           (optional, for evaluation)
    """

    def __init__(self, seq_dir: Path, cfg: PipelineCfg) -> None:
        self._dir = seq_dir
        self._cfg = cfg

        # ── parse seqinfo.ini ── #
        ini = configparser.ConfigParser()
        ini_path = seq_dir / "seqinfo.ini"
        if not ini_path.exists():
            raise FileNotFoundError(f"seqinfo.ini not found in {seq_dir}")
        ini.read(ini_path)
        seq = ini["Sequence"]

        self._fps = float(seq.get("frameRate", 25))
        self._seq_len = int(seq.get("seqLength", 0))
        self._w = int(seq.get("imWidth", 0))
        self._h = int(seq.get("imHeight", 0))
        img_dir_name = seq.get("imDir", "img1")
        img_ext = seq.get("imExt", ".jpg")

        self._img_dir = seq_dir / img_dir_name
        if not self._img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self._img_dir}")

        self._images: List[Path] = sorted(
            p for p in self._img_dir.iterdir()
            if p.suffix.lower() == img_ext.lower()
        )
        if not self._images:
            raise ValueError(f"No {img_ext} images found in {self._img_dir}")

        # ── load det.txt if present ── #
        self._det_map: dict[int, np.ndarray] = {}   # frame_id (1-based) → (N,5)
        det_path = seq_dir / "det" / "det.txt"
        if det_path.exists():
            self._det_map = _load_mot_det(det_path)
            logger.info("Loaded pre-computed detections from %s", det_path)

        logger.info(
            "MOT17Ingestion: %s  (%dx%d @ %.1f fps, %d frames)",
            seq_dir.name, self._w, self._h, self._fps, len(self._images),
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return len(self._images)

    @property
    def resolution(self) -> tuple[int, int]:
        return self._w, self._h

    def __iter__(self) -> Iterator[RawFrame]:
        sample = self._cfg.frame_sample_rate
        source = str(self._dir)

        for global_idx, img_path in enumerate(self._images):
            if global_idx % sample != 0:
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Could not read image: %s", img_path)
                continue
            # MOT frame IDs are 1-based
            mot_frame_id = global_idx + 1
            precomputed = self._det_map.get(mot_frame_id)
            timestamp = global_idx / self._fps
            yield RawFrame(
                frame_idx=global_idx,
                timestamp=timestamp,
                image=frame,
                source=source,
                fps=self._fps,
                precomputed_dets=precomputed,
            )


# ══════════════════════════════════════════════════════════════════════════ #
#  Static image folder
# ══════════════════════════════════════════════════════════════════════════ #

class ImageIngestion(BaseIngestion):
    """
    Treats every image in a directory as an independent frame.
    No temporal continuity — tracking is disabled for this mode.
    """

    def __init__(self, folder: Path, cfg: PipelineCfg) -> None:
        self._folder = folder
        self._cfg = cfg
        self._images: List[Path] = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )
        if not self._images:
            raise ValueError(f"No images found in {folder}")
        logger.info("ImageIngestion: %d images in %s", len(self._images), folder)

    @property
    def fps(self) -> float:
        return 1.0   # not meaningful for static images

    @property
    def total_frames(self) -> int:
        return len(self._images)

    @property
    def resolution(self) -> tuple[int, int]:
        img = cv2.imread(str(self._images[0]))
        if img is not None:
            return img.shape[1], img.shape[0]
        return 0, 0

    def __iter__(self) -> Iterator[RawFrame]:
        for idx, img_path in enumerate(self._images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Could not read image: %s", img_path)
                continue
            yield RawFrame(
                frame_idx=idx,
                timestamp=float(idx),
                image=frame,
                source=str(img_path),
                fps=self._fps,
            )

    @property
    def _fps(self) -> float:
        return 1.0


# ══════════════════════════════════════════════════════════════════════════ #
#  Factory
# ══════════════════════════════════════════════════════════════════════════ #

def open_source(
    input_path: Path,
    cfg: PipelineCfg,
    mode: str = "auto",
) -> tuple[BaseIngestion, str]:
    """
    Return (ingestion_object, effective_mode).

    *mode* is one of "auto" | "video" | "sequence" | "images".
    When "auto", the type is inferred from the filesystem layout.
    """
    p = input_path.resolve()

    if mode == "auto":
        if p.is_file():
            mode = "video"
        elif (p / "seqinfo.ini").exists():
            mode = "sequence"
        elif p.is_dir():
            mode = "images"
        else:
            raise ValueError(f"Cannot determine input mode for: {p}")

    if mode == "video":
        return VideoIngestion(p, cfg), "video"
    if mode == "sequence":
        return MOT17Ingestion(p, cfg), "sequence"
    if mode == "images":
        return ImageIngestion(p, cfg), "images"

    raise ValueError(f"Unknown mode: {mode}")


# ══════════════════════════════════════════════════════════════════════════ #
#  Background reader thread
# ══════════════════════════════════════════════════════════════════════════ #

_SENTINEL = None   # signals downstream threads to stop

# ══════════════════════════════════════════════════════════════════════════ #
#  Background reader thread
# ══════════════════════════════════════════════════════════════════════════ #

_SENTINEL = None   # signals downstream threads to stop


class ReaderThread(threading.Thread):
    """
    Reads frames from *source* and puts them into *out_q*.
    Puts a None sentinel when the source is exhausted or stop() is called.
    """

    def __init__(
        self,
        source: BaseIngestion,
        out_q: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="ReaderThread", daemon=True)
        self._source = source
        self._out_q = out_q
        self._stop_event = stop_event  # FIXED: Renamed from self._stop
        self.frames_read: int = 0
        self.exception: Optional[Exception] = None

    def run(self) -> None:
        try:
            for raw_frame in self._source:
                if self._stop_event.is_set():  # FIXED
                    break
                # Block until there is space; check stop every 0.1 s
                while True:
                    try:
                        self._out_q.put(raw_frame, timeout=0.1)
                        self.frames_read += 1  # (Optional) incrementing so the counter works
                        break
                    except queue.Full:
                        if self._stop_event.is_set():  # FIXED
                            return
        except Exception as exc:
            logger.exception("ReaderThread crashed: %s", exc)
            self.exception = exc
        finally:
            # Always send sentinel so downstream threads can exit
            self._out_q.put(_SENTINEL)
            logger.debug("ReaderThread finished (%d frames read)", self.frames_read)
'''
class ReaderThread(threading.Thread):
    """
    Reads frames from *source* and puts them into *out_q*.
    Puts a None sentinel when the source is exhausted or stop() is called.
    """

    def __init__(
        self,
        source: BaseIngestion,
        out_q: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="ReaderThread", daemon=True)
        self._source = source
        self._out_q = out_q
        self._stop = stop_event
        self.frames_read: int = 0
        self.exception: Optional[Exception] = None

    def run(self) -> None:
        try:
            for raw_frame in self._source:
                if self._stop.is_set():
                    break
                # Block until there is space; check stop every 0.1 s
                while True:
                    try:
                        self._out_q.put(raw_frame, timeout=0.1)
                        break
                    except queue.Full:
                        if self._stop.is_set():
                            return
        except Exception as exc:
            logger.exception("ReaderThread crashed: %s", exc)
            self.exception = exc
        finally:
            # Always send sentinel so downstream threads can exit
            self._out_q.put(_SENTINEL)
            logger.debug("ReaderThread finished (%d frames read)", self.frames_read)
'''

# ══════════════════════════════════════════════════════════════════════════ #
#  Private helpers
# ══════════════════════════════════════════════════════════════════════════ #

def _load_mot_det(det_path: Path) -> dict[int, np.ndarray]:
    """
    Parse MOT det.txt format:
        frame_id, -1, x, y, w, h, conf, -1, -1, -1
    Returns {frame_id: np.ndarray shape (N,5) → [x, y, w, h, conf]}.
    """
    from collections import defaultdict
    rows: dict[int, list] = defaultdict(list)
    with det_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                fid = int(parts[0])
                x, y, w, h, conf = (
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                )
                rows[fid].append([x, y, w, h, conf])
            except ValueError:
                continue
    return {fid: np.array(v, dtype=np.float32) for fid, v in rows.items()}