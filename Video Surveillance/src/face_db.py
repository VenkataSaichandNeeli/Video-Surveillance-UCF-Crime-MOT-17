"""
Persistent face-identity database backed by SQLite.

Design goals
------------
* Permanent IDs survive program restarts, dataset changes, and year-long gaps.
* Thread-safe: a single connection is shared under a Lock.
* Efficient similarity search via fully-vectorised NumPy dot-product.
* Graceful fall-back to body-appearance embeddings when face is absent.

Schema
------
identities
    permanent_id  INTEGER PK AUTOINCREMENT
    face_emb      BLOB (float32 x 512, L2-normalised)
    body_emb      BLOB (optional float32 x 512)
    first_seen    REAL  (Unix timestamp)
    last_seen     REAL
    appearances   INTEGER DEFAULT 1
    source        TEXT   (first seen in which video)
    thumbnail     TEXT   (path to a small JPEG of the face)

appearances
    id            INTEGER PK AUTOINCREMENT
    pid           INTEGER REFERENCES identities(permanent_id)
    source        TEXT
    frame_num     INTEGER
    ts            REAL
    bbox          TEXT   (JSON "[x1,y1,x2,y2]")
    conf          REAL
    activity      TEXT
    anomaly       TEXT
    logged_at     REAL
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineCfg

logger = logging.getLogger(__name__)

_EMB_DTYPE = np.float32
_EMB_DIM = 512


class FaceDB:
    """
    Thread-safe persistent identity store.

    All public methods acquire `_lock` internally, so callers need not
    worry about concurrent access.
    """

    def __init__(self, cfg: PipelineCfg) -> None:
        self._cfg = cfg
        db_path = Path(cfg.face_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._open(db_path)
        self._create_schema()
        logger.info("FaceDB opened: %s", db_path)

    # ── lifecycle ─────────────────────────────────────────────────────── #

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
        logger.debug("FaceDB closed")

    def __enter__(self) -> FaceDB:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── public API ────────────────────────────────────────────────────── #

    def find_or_create(
        self,
        face_emb: Optional[np.ndarray],
        body_emb: Optional[np.ndarray],
        source: str,
        timestamp: float,
    ) -> Tuple[int, float]:
        """
        Search for an existing identity matching *face_emb* (or *body_emb*
        if face_emb is None).  Create a new record if no match found.

        Returns
        -------
        permanent_id : int
        similarity   : float   (0.0 if newly created)
        """
        query_emb = face_emb if face_emb is not None else body_emb
        if query_emb is None:
            return self._new_identity(None, None, source, timestamp), 0.0

        match = self._search(query_emb, use_face=(face_emb is not None))
        if match is not None:
            pid, sim = match
            self._update_last_seen(pid, source, timestamp, body_emb)
            return pid, sim

        return self._new_identity(face_emb, body_emb, source, timestamp), 0.0

    def log_appearance(
        self,
        permanent_id: int,
        source: str,
        frame_num: int,
        timestamp: float,
        bbox: List[float],
        confidence: float,
        activity: str = "",
        anomaly: str = "",
    ) -> None:
        """Insert one row into the appearances table."""
        bbox_str = json.dumps([round(float(v), 1) for v in bbox])
        with self._lock:
            self._conn.execute(
                """INSERT INTO appearances
                   (pid, source, frame_num, ts, bbox, conf, activity, anomaly, logged_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    permanent_id, source, frame_num, timestamp,
                    bbox_str, confidence, activity, anomaly,
                    time.time(),
                ),
            )
            self._conn.commit()

    def save_thumbnail(self, permanent_id: int, thumb_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE identities SET thumbnail=? WHERE permanent_id=?",
                (thumb_path, permanent_id),
            )
            self._conn.commit()

    def total_identities(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM identities")
            return cur.fetchone()[0]

    # ── internals ─────────────────────────────────────────────────────── #

    @staticmethod
    def _open(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32000")   # 32 MB page cache
        return conn

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    permanent_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_emb      BLOB,
                    body_emb      BLOB,
                    first_seen    REAL    NOT NULL,
                    last_seen     REAL    NOT NULL,
                    appearances   INTEGER NOT NULL DEFAULT 1,
                    source        TEXT,
                    thumbnail     TEXT
                );
                CREATE TABLE IF NOT EXISTS appearances (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    pid         INTEGER NOT NULL REFERENCES identities(permanent_id),
                    source      TEXT    NOT NULL,
                    frame_num   INTEGER NOT NULL,
                    ts          REAL    NOT NULL,
                    bbox        TEXT,
                    conf        REAL,
                    activity    TEXT,
                    anomaly     TEXT,
                    logged_at   REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_appearances_pid ON appearances(pid);
                """
            )
            self._conn.commit()

    def _search(
        self,
        query_emb: np.ndarray,
        use_face: bool,
    ) -> Optional[Tuple[int, float]]:
        """
        Vectorised cosine-similarity search.
        Both stored and query embeddings are L2-normalised, so
        cosine_similarity = dot_product.

        Returns (permanent_id, similarity) or None.
        """
        col = "face_emb" if use_face else "body_emb"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT permanent_id, {col} FROM identities WHERE {col} IS NOT NULL"
            ).fetchall()

        if not rows:
            return None

        ids: List[int] = []
        embs: List[np.ndarray] = []
        for pid, blob in rows:
            if blob is None:
                continue
            emb = np.frombuffer(blob, dtype=_EMB_DTYPE).copy()
            n = np.linalg.norm(emb)
            if n < 1e-6:
                continue
            ids.append(pid)
            embs.append(emb / n)

        if not ids:
            return None

        matrix = np.stack(embs)                    # (N, 512)
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        sims = matrix @ q                           # (N,)
        best = int(np.argmax(sims))
        best_sim = float(sims[best])

        if best_sim >= self._cfg.face.embedding_similarity_threshold:
            return ids[best], best_sim

        return None

    def _new_identity(
        self,
        face_emb: Optional[np.ndarray],
        body_emb: Optional[np.ndarray],
        source: str,
        timestamp: float,
    ) -> int:
        face_blob = face_emb.astype(_EMB_DTYPE).tobytes() if face_emb is not None else None
        body_blob = body_emb.astype(_EMB_DTYPE).tobytes() if body_emb is not None else None
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO identities
                   (face_emb, body_emb, first_seen, last_seen, appearances, source)
                   VALUES (?,?,?,?,1,?)""",
                (face_blob, body_blob, timestamp, timestamp, source),
            )
            self._conn.commit()
            pid = cur.lastrowid
        logger.debug("New identity created: permanent_id=%d", pid)
        return pid

    def _update_last_seen(
        self,
        pid: int,
        source: str,
        timestamp: float,
        body_emb: Optional[np.ndarray],
    ) -> None:
        body_blob = body_emb.astype(_EMB_DTYPE).tobytes() if body_emb is not None else None
        with self._lock:
            if body_blob is not None:
                self._conn.execute(
                    """UPDATE identities
                       SET last_seen=?, appearances=appearances+1, body_emb=?
                       WHERE permanent_id=?""",
                    (timestamp, body_blob, pid),
                )
            else:
                self._conn.execute(
                    """UPDATE identities
                       SET last_seen=?, appearances=appearances+1
                       WHERE permanent_id=?""",
                    (timestamp, pid),
                )
            self._conn.commit()