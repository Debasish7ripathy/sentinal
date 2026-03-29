"""
Auto-recording service for SENTINEL.
Records video clips on threat detection using OpenCV VideoWriter.
"""
import os
import time
import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from sentinel.config import settings
from sentinel.models.database import SessionLocal, Recording


class RecordingManager:
    """Manages auto-recording of threat events."""

    def __init__(self):
        self.is_recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.recording_start = 0
        self.current_filename = ""
        self.current_path = ""
        self.current_threat_level = "NORMAL"
        self.current_weapon = False
        self.normal_since: Optional[float] = None
        self.frame_size = (640, 480)
        self.recordings_dir = Path(settings.RECORDINGS_DIR)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    def update(self, frame: np.ndarray, threat_level: str, weapon_detected: bool):
        """Called each frame to manage recording state."""
        should_record = threat_level in ("SUSPICIOUS", "CRITICAL")

        if should_record and not self.is_recording:
            self._start(frame, threat_level, weapon_detected)

        if self.is_recording:
            self._write_frame(frame)

            elapsed = time.time() - self.recording_start
            normal_dur = (time.time() - self.normal_since) if self.normal_since else 0

            # Stop: 10s min + normal for 5s, or 30s max
            if (elapsed > settings.RECORDING_DURATION_SEC and normal_dur > 5) or elapsed > 30:
                self._stop()

        # Track normal state
        if threat_level == "NORMAL":
            if self.normal_since is None:
                self.normal_since = time.time()
        else:
            self.normal_since = None

    def _start(self, frame: np.ndarray, threat_level: str, weapon_detected: bool):
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        self.current_threat_level = threat_level
        self.current_weapon = weapon_detected

        now = datetime.datetime.now()
        time_str = now.strftime("%H-%M-%S")
        prefix = "WEAPON_" if weapon_detected else ""
        self.current_filename = f"sentinel_{prefix}{threat_level}_{time_str}.avi"
        self.current_path = str(self.recordings_dir / self.current_filename)

        fourcc = cv2.VideoWriter_fourcc(*settings.RECORDING_CODEC)
        self.writer = cv2.VideoWriter(self.current_path, fourcc, 20.0, (w, h))
        self.recording_start = time.time()
        self.is_recording = True

    def _write_frame(self, frame: np.ndarray):
        if self.writer and self.writer.isOpened():
            self.writer.write(frame)

    def _stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None

        duration = time.time() - self.recording_start
        file_size = 0
        if os.path.exists(self.current_path):
            file_size = os.path.getsize(self.current_path)

        # Save to database
        db = SessionLocal()
        try:
            rec = Recording(
                filename=self.current_filename,
                filepath=self.current_path,
                duration_sec=round(duration, 1),
                threat_level=self.current_threat_level,
                weapon_detected=self.current_weapon,
                file_size_bytes=file_size,
            )
            db.add(rec)
            db.commit()

            # Enforce max recordings
            total = db.query(Recording).count()
            if total > settings.MAX_RECORDINGS:
                oldest = db.query(Recording).order_by(Recording.timestamp.asc()).first()
                if oldest:
                    try:
                        os.remove(oldest.filepath)
                    except OSError:
                        pass
                    db.delete(oldest)
                    db.commit()
        finally:
            db.close()

        self.is_recording = False
        self.current_filename = ""
        self.current_path = ""

    def get_recordings(self) -> list:
        db = SessionLocal()
        try:
            recs = db.query(Recording).order_by(Recording.timestamp.desc()).limit(20).all()
            return [r.to_dict() for r in recs]
        finally:
            db.close()

    def release(self):
        if self.is_recording:
            self._stop()
