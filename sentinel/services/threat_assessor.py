"""
Threat Assessment State Machine for SENTINEL.

Combines pose analysis, object detection, velocity tracking,
and ML model outputs into a unified threat level + confidence score.
"""
import time
from collections import deque


class ThreatAssessor:
    """Threat classification state machine."""

    def __init__(self):
        self.person_in_frame_since = None
        self.previous_keypoints = None
        self.previous_keypoint_time = 0
        self.keypoint_velocities = deque(maxlen=30)
        self.sustained_aggressive_since = None
        self.sensitivity_multiplier = 1.0

    def set_sensitivity(self, level: str):
        self.sensitivity_multiplier = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.3,
        }.get(level, 1.0)

    def assess(self, person_detected: bool, person_bbox: dict,
               landmarks, objects: list, weapon_detected: bool,
               held_weapon: bool, ml_probs: dict, anomaly_score: float,
               bbox_history: deque, frame_size: tuple) -> dict:
        """
        Assess current threat level.

        Returns dict with: level, confidence, proximity_score, velocity_score,
        posture_score, weapon_score, follower_score, details
        """
        w, h = frame_size

        if not person_detected:
            self.person_in_frame_since = None
            self.previous_keypoints = None
            self.keypoint_velocities.clear()
            self.sustained_aggressive_since = None
            return {
                "level": "NORMAL",
                "confidence": 0,
                "proximity_score": 0,
                "velocity_score": 0,
                "posture_score": 0,
                "weapon_score": 0,
                "follower_score": 0,
                "details": "No person detected",
            }

        if self.person_in_frame_since is None:
            self.person_in_frame_since = time.time()

        # ---- Proximity Score (0-100) ----
        proximity_score = 0
        if person_bbox:
            coverage = (person_bbox["w"] * person_bbox["h"]) / (w * h) * 100
            proximity_score = min(100, coverage * 2)

            # Track bbox for approach detection
            bbox_history.append({"size": coverage, "time": time.time()})

            # Check rapid approach
            if len(bbox_history) > 10:
                recent = list(bbox_history)[-10:]
                oldest = recent[0]["size"]
                newest = recent[-1]["size"]
                dt = recent[-1]["time"] - recent[0]["time"]
                if dt > 0 and oldest > 0:
                    growth_rate = (newest - oldest) / oldest
                    if growth_rate > 0.4 and dt < 2.0:
                        proximity_score = 100

        # ---- Velocity Score (0-100) ----
        velocity_score = 0
        if landmarks:
            now = time.time()
            current_kp = [(lm.x * w, lm.y * h) for lm in landmarks]
            if self.previous_keypoints:
                dt = now - self.previous_keypoint_time
                if dt > 0:
                    total_vel = 0
                    count = 0
                    for curr, prev in zip(current_kp, self.previous_keypoints):
                        dx = curr[0] - prev[0]
                        dy = curr[1] - prev[1]
                        total_vel += (dx**2 + dy**2)**0.5 / dt
                        count += 1
                    avg_vel = total_vel / max(count, 1)
                    self.keypoint_velocities.append(avg_vel)

            self.previous_keypoints = current_kp
            self.previous_keypoint_time = now

            if len(self.keypoint_velocities) > 5:
                recent_vel = list(self.keypoint_velocities)[-5:]
                avg = sum(recent_vel) / len(recent_vel)
                velocity_score = min(100, avg / 3)

        # ---- Posture Score (0-100) ----
        posture_score = 0
        if landmarks:
            nose = landmarks[0]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]

            # Both wrists above nose = aggressive
            if l_wrist.y < nose.y and r_wrist.y < nose.y:
                posture_score = 100
                if self.sustained_aggressive_since is None:
                    self.sustained_aggressive_since = time.time()
            elif l_wrist.y < l_shoulder.y or r_wrist.y < r_shoulder.y:
                posture_score = 50
                self.sustained_aggressive_since = None
            else:
                self.sustained_aggressive_since = None

            # Loitering detection
            if self.person_in_frame_since:
                in_frame_sec = time.time() - self.person_in_frame_since
                if in_frame_sec > 8.0 and velocity_score < 20:
                    posture_score = max(posture_score, 40)

        # ---- Weapon Score (0-100) ----
        weapon_score = 0
        if weapon_detected:
            weapon_score = 100
            if held_weapon:
                weapon_score = 100

        # ---- ML Boost ----
        ml_boost = 0
        if ml_probs:
            if ml_probs.get("critical", 0) > 0.6:
                ml_boost = 15
            elif ml_probs.get("suspicious", 0) > 0.6:
                ml_boost = 8

        if anomaly_score > 0.01:
            ml_boost += min(10, anomaly_score * 500)

        # ---- Weighted Confidence ----
        raw_confidence = (
            proximity_score * 0.20 +
            velocity_score * 0.20 +
            posture_score * 0.20 +
            weapon_score * 0.30 +
            ml_boost
        )
        confidence = min(100, int(raw_confidence * self.sensitivity_multiplier))

        # ---- Determine Level ----
        level = "NORMAL"
        details = []

        if weapon_detected:
            level = "CRITICAL"
            details.append(f"Weapon detected{' (HELD)' if held_weapon else ''}")
        elif confidence >= 60:
            level = "CRITICAL"
        elif person_bbox and (person_bbox["w"] * person_bbox["h"]) / (w * h) > 0.5:
            level = "CRITICAL"
            details.append("Person covering >50% of frame")
        elif posture_score >= 100 and velocity_score > 40:
            level = "CRITICAL"
            details.append("Aggressive posture + high velocity")
        elif self.sustained_aggressive_since and (time.time() - self.sustained_aggressive_since) > 1.5:
            level = "CRITICAL"
            details.append("Sustained aggressive posture >1.5s")
        elif confidence >= 25:
            level = "SUSPICIOUS"
        elif posture_score >= 40:
            level = "SUSPICIOUS"
            details.append("Unusual posture")
        elif proximity_score >= 50:
            level = "SUSPICIOUS"
            details.append("Close proximity")

        if ml_probs and ml_probs.get("critical", 0) > 0.7:
            if level != "CRITICAL":
                level = "CRITICAL"
                details.append("ML model: high critical probability")

        return {
            "level": level,
            "confidence": confidence,
            "proximity_score": round(proximity_score, 1),
            "velocity_score": round(velocity_score, 1),
            "posture_score": round(posture_score, 1),
            "weapon_score": round(weapon_score, 1),
            "follower_score": 0,
            "ml_boost": round(ml_boost, 1),
            "details": "; ".join(details) if details else "Monitoring",
        }
