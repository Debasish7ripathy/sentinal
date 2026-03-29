"""SQLAlchemy database models and engine setup for SENTINEL."""
import datetime
import json
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Text,
    DateTime, JSON, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from sentinel.config import settings

Base = declarative_base()


class Follower(Base):
    """Persistent multi-day follower tracking."""
    __tablename__ = "followers"

    id = Column(String(50), primary_key=True)
    fingerprint = Column(String(100), index=True)
    first_seen = Column(DateTime, default=datetime.datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    total_appearances = Column(Integer, default=1)
    appearances_by_day = Column(JSON, default=dict)
    total_time_in_frame = Column(Float, default=0.0)
    avg_approach_speed = Column(String(20), default="unknown")
    max_proximity = Column(Float, default=0.0)
    threat_level_history = Column(JSON, default=list)
    objects_detected = Column(JSON, default=list)
    notes = Column(Text, default="")
    risk_score = Column(Integer, default=0)
    thumbnail_path = Column(String(500), default="")
    marked_safe = Column(Boolean, default=False)
    embedding = Column(Text, default="")  # JSON-serialized re-id embedding vector

    incidents = relationship("Incident", back_populates="follower")

    __table_args__ = (
        Index("idx_risk_score", "risk_score"),
        Index("idx_last_seen", "last_seen"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "firstSeen": self.first_seen.isoformat() if self.first_seen else None,
            "lastSeen": self.last_seen.isoformat() if self.last_seen else None,
            "totalAppearances": self.total_appearances,
            "appearancesByDay": self.appearances_by_day or {},
            "totalTimeInFrame": self.total_time_in_frame,
            "avgApproachSpeed": self.avg_approach_speed,
            "maxProximity": self.max_proximity,
            "threatLevelHistory": self.threat_level_history or [],
            "objectsDetected": self.objects_detected or [],
            "notes": self.notes,
            "riskScore": self.risk_score,
            "thumbnailPath": self.thumbnail_path,
            "markedSafe": self.marked_safe,
        }

    def calculate_risk_score(self):
        apps = self.total_appearances or 0
        days_data = self.appearances_by_day or {}
        sorted_days = sorted(days_data.keys())

        consecutive_days = 1
        for i in range(len(sorted_days) - 1, 0, -1):
            d1 = datetime.date.fromisoformat(sorted_days[i])
            d2 = datetime.date.fromisoformat(sorted_days[i - 1])
            if (d1 - d2).days <= 1:
                consecutive_days += 1
            else:
                break

        weapon_keywords = {"bat", "knife", "baseball bat", "BAT DETECTED", "KNIFE DETECTED"}
        weapon_detected = bool(
            set(self.objects_detected or []) & weapon_keywords
        )

        score = (
            (30 if apps > 5 else apps * 6)
            + (consecutive_days * 15)
            + (35 if weapon_detected else 0)
            + (10 if self.avg_approach_speed == "fast" else 0)
            + (10 if (self.max_proximity or 0) > 60 else 0)
        )
        self.risk_score = min(100, score)
        return self.risk_score


class Incident(Base):
    """Incident / event log entries."""
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    level = Column(String(20), index=True)  # NORMAL, SUSPICIOUS, CRITICAL, WEAPON, INFO, ERROR
    message = Column(Text)
    confidence = Column(Float, default=0.0)
    threat_type = Column(String(50), default="")
    objects_detected = Column(JSON, default=list)
    follower_id = Column(String(50), ForeignKey("followers.id"), nullable=True)
    recording_path = Column(String(500), default="")
    frame_snapshot_path = Column(String(500), default="")

    follower = relationship("Follower", back_populates="incidents")

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "message": self.message,
            "confidence": self.confidence,
            "threatType": self.threat_type,
            "objectsDetected": self.objects_detected or [],
            "followerId": self.follower_id,
            "recordingPath": self.recording_path,
        }


class Recording(Base):
    """Auto-recorded threat clips."""
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), unique=True)
    filepath = Column(String(500))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    duration_sec = Column(Float, default=0.0)
    threat_level = Column(String(20))
    weapon_detected = Column(Boolean, default=False)
    file_size_bytes = Column(Integer, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "durationSec": self.duration_sec,
            "threatLevel": self.threat_level,
            "weaponDetected": self.weapon_detected,
            "fileSizeBytes": self.file_size_bytes,
        }


class ThreatEvent(Base):
    """Continuous threat state log for ML training data collection."""
    __tablename__ = "threat_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    threat_level = Column(String(20))
    confidence = Column(Float)
    proximity_score = Column(Float)
    velocity_score = Column(Float)
    posture_score = Column(Float)
    weapon_score = Column(Float)
    follower_score = Column(Float)
    num_persons = Column(Integer, default=0)
    objects_in_frame = Column(JSON, default=list)
    pose_landmarks = Column(Text, default="")  # JSON keypoints
    bbox_coverage = Column(Float, default=0.0)


class TrainingLabel(Base):
    """Human-annotated labels for retraining."""
    __tablename__ = "training_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    frame_path = Column(String(500))
    label = Column(String(50))  # normal, suspicious, critical
    pose_data = Column(Text, default="")
    objects_data = Column(Text, default="")
    annotator = Column(String(100), default="auto")


# --- Engine & Session ---

engine = create_engine(
    f"sqlite:///{settings.DB_PATH}",
    echo=settings.DEBUG,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
