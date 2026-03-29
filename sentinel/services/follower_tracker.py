"""
Persistent multi-day follower tracking for SENTINEL.

Tracks repeat appearances across sessions using:
- Time-of-day pattern matching
- Approach direction fingerprinting
- Body proportion (bbox aspect ratio) bucketing
- Optional Re-ID embedding similarity (when model is trained)
"""
import datetime
import json
import hashlib
from typing import Optional

import numpy as np

from sentinel.config import settings
from sentinel.models.database import SessionLocal, Follower


class FollowerTracker:
    """Manage follower identification and tracking."""

    def __init__(self):
        self.session_notified = set()
        self.repeat_notified = set()
        self.armed_notified = set()
        self.current_follower_id = None

    def track(self, person_bbox: dict, frame_size: tuple,
              threat_level: str, objects: list,
              reid_embedding: Optional[list] = None,
              weapon_detected: bool = False,
              weapon_label: str = "") -> dict:
        """
        Track a detected person, match to existing follower or create new.

        Returns dict with follower info and alert status.
        """
        w, h = frame_size
        now = datetime.datetime.utcnow()
        today = now.strftime("%Y-%m-%d")

        fingerprint = self._generate_fingerprint(person_bbox, w)

        db = SessionLocal()
        try:
            # Try to match existing follower
            match_id = self._find_match(db, fingerprint, reid_embedding)

            if match_id:
                follower = db.query(Follower).get(match_id)
                result = self._update_follower(db, follower, today, now, person_bbox,
                                                w, h, threat_level, objects,
                                                weapon_detected, weapon_label, reid_embedding)
            else:
                follower, result = self._create_follower(db, fingerprint, today, now,
                                                          person_bbox, w, h, threat_level,
                                                          reid_embedding)

            self.current_follower_id = follower.id
            db.commit()

            # Generate alerts
            alerts = self._check_alerts(follower, today, weapon_detected, weapon_label)
            result["alerts"] = alerts

            return result
        finally:
            db.close()

    def _generate_fingerprint(self, bbox: dict, frame_width: float) -> str:
        now = datetime.datetime.now()
        time_bucket = now.hour // 2

        center_x = bbox["x"] + bbox["w"] / 2
        if center_x < frame_width * 0.33:
            approach = "L"
        elif center_x > frame_width * 0.66:
            approach = "R"
        else:
            approach = "C"

        aspect = bbox["h"] / max(1, bbox["w"])
        if aspect < 2:
            size_bucket = "S"
        elif aspect < 3:
            size_bucket = "M"
        else:
            size_bucket = "L"

        return f"T{time_bucket}_{approach}_{size_bucket}"

    def _fingerprint_similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        parts_a = a.split("_")
        parts_b = b.split("_")
        if len(parts_a) != 3 or len(parts_b) != 3:
            return 0.0

        score = 0.0
        # Time bucket
        t_a = int(parts_a[0][1:])
        t_b = int(parts_b[0][1:])
        if t_a == t_b:
            score += 0.40
        elif abs(t_a - t_b) <= 1:
            score += 0.25

        # Direction
        if parts_a[1] == parts_b[1]:
            score += 0.35

        # Size
        if parts_a[2] == parts_b[2]:
            score += 0.25

        return score

    def _embedding_similarity(self, emb_a: list, emb_b: list) -> float:
        """Cosine similarity between two Re-ID embeddings."""
        a = np.array(emb_a)
        b = np.array(emb_b)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _find_match(self, db, fingerprint: str, reid_embedding: Optional[list]) -> Optional[str]:
        followers = db.query(Follower).filter(Follower.marked_safe == False).all()

        best_match = None
        best_score = 0.0

        for f in followers:
            # Fingerprint similarity
            fp_sim = self._fingerprint_similarity(fingerprint, f.fingerprint or "")

            # Re-ID embedding similarity (if available)
            emb_sim = 0.0
            if reid_embedding and f.embedding:
                try:
                    stored_emb = json.loads(f.embedding)
                    emb_sim = self._embedding_similarity(reid_embedding, stored_emb)
                except (json.JSONDecodeError, ValueError):
                    pass

            # Combined score (weight embedding more if available)
            if emb_sim > 0:
                combined = fp_sim * 0.4 + emb_sim * 0.6
            else:
                combined = fp_sim

            if combined > best_score and combined >= settings.FINGERPRINT_MATCH_THRESHOLD:
                best_score = combined
                best_match = f.id

        return best_match

    def _update_follower(self, db, follower: Follower, today: str,
                          now, bbox, w, h, threat_level, objects,
                          weapon_detected, weapon_label, reid_embedding) -> dict:
        follower.last_seen = now
        follower.total_appearances += 1

        days = follower.appearances_by_day or {}
        days[today] = days.get(today, 0) + 1
        follower.appearances_by_day = days

        follower.total_time_in_frame += 0.1  # approx per call

        coverage = (bbox["w"] * bbox["h"]) / (w * h) * 100
        follower.max_proximity = max(follower.max_proximity or 0, coverage)

        # Update threat history
        history = follower.threat_level_history or []
        if threat_level.lower() not in history:
            history.append(threat_level.lower())
            follower.threat_level_history = history

        # Update objects
        obj_list = follower.objects_detected or []
        if weapon_detected and weapon_label and weapon_label not in obj_list:
            obj_list.append(weapon_label)
        for obj in objects:
            if obj not in obj_list:
                obj_list.append(obj)
        follower.objects_detected = obj_list

        # Update embedding
        if reid_embedding:
            follower.embedding = json.dumps(reid_embedding)

        # Recalculate risk
        follower.calculate_risk_score()

        # Update notes
        days_count = len(follower.appearances_by_day)
        follower.notes = f"Appeared {days_count} day(s). " + (
            f"Objects: {', '.join(obj_list)}." if obj_list else ""
        )

        return follower.to_dict()

    def _create_follower(self, db, fingerprint, today, now, bbox, w, h,
                          threat_level, reid_embedding) -> tuple:
        count = db.query(Follower).count()
        fid = f"follower_{str(count + 1).zfill(3)}"

        coverage = (bbox["w"] * bbox["h"]) / (w * h) * 100

        follower = Follower(
            id=fid,
            fingerprint=fingerprint,
            first_seen=now,
            last_seen=now,
            total_appearances=1,
            appearances_by_day={today: 1},
            total_time_in_frame=0,
            avg_approach_speed="unknown",
            max_proximity=coverage,
            threat_level_history=[threat_level.lower()],
            objects_detected=[],
            notes="",
            risk_score=0,
            embedding=json.dumps(reid_embedding) if reid_embedding else "",
        )
        follower.calculate_risk_score()
        db.add(follower)

        return follower, follower.to_dict()

    def _check_alerts(self, follower: Follower, today: str,
                       weapon_detected: bool, weapon_label: str) -> list:
        alerts = []
        days_data = follower.appearances_by_day or {}
        days_count = len(days_data)
        today_apps = days_data.get(today, 0)

        # Consecutive days
        sorted_days = sorted(days_data.keys())
        consecutive_days = 1
        for i in range(len(sorted_days) - 1, 0, -1):
            d1 = datetime.date.fromisoformat(sorted_days[i])
            d2 = datetime.date.fromisoformat(sorted_days[i-1])
            if (d1 - d2).days <= 1:
                consecutive_days += 1
            else:
                break

        # Alert: Armed known threat (immediate)
        if weapon_detected and follower.id not in self.armed_notified:
            alerts.append({
                "type": "ARMED_KNOWN_THREAT",
                "level": "CRITICAL",
                "message": f"ARMED KNOWN THREAT: {follower.id} with {weapon_label}. Risk: {follower.risk_score}",
                "priority": "urgent",
                "bypass_cooldown": True,
            })
            self.armed_notified.add(follower.id)

        # Alert: Confirmed follower (3+ consecutive days)
        elif consecutive_days >= 3 and follower.id not in self.session_notified:
            alerts.append({
                "type": "CONFIRMED_FOLLOWER",
                "level": "CRITICAL",
                "message": f"CONFIRMED FOLLOWER: {follower.id}. {consecutive_days} consecutive days. Risk: {follower.risk_score}",
                "priority": "urgent",
            })
            self.session_notified.add(follower.id)

        # Alert: Possible follower (2+ different days)
        elif days_count >= 2 and follower.id not in self.session_notified:
            alerts.append({
                "type": "POSSIBLE_FOLLOWER",
                "level": "SUSPICIOUS",
                "message": f"POSSIBLE FOLLOWER: {follower.id}. Seen on {days_count} different days.",
                "priority": "high",
            })
            self.session_notified.add(follower.id)

        # Alert: Repeat presence same day
        elif today_apps >= 2 and follower.id not in self.repeat_notified:
            alerts.append({
                "type": "REPEAT_PRESENCE",
                "level": "SUSPICIOUS",
                "message": f"REPEAT PRESENCE: {follower.id}. {today_apps}x today.",
                "priority": "default",
            })
            self.repeat_notified.add(follower.id)

        return alerts

    def get_all_followers(self) -> list:
        db = SessionLocal()
        try:
            followers = db.query(Follower).filter(
                Follower.marked_safe == False
            ).order_by(Follower.risk_score.desc()).all()
            return [f.to_dict() for f in followers]
        finally:
            db.close()

    def mark_safe(self, follower_id: str) -> bool:
        db = SessionLocal()
        try:
            f = db.query(Follower).get(follower_id)
            if f:
                f.marked_safe = True
                db.commit()
                return True
            return False
        finally:
            db.close()
