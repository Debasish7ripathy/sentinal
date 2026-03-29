"""
Real-time inference engine for SENTINEL.

Orchestrates:
- OpenCV camera capture
- MediaPipe pose detection
- YOLOv8 object/weapon detection
- ML model inference (threat classifier, anomaly, re-id)
- Frame annotation and streaming
"""
import time
import json
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import torch

from sentinel.config import settings
from sentinel.services.threat_assessor import ThreatAssessor
from sentinel.services.follower_tracker import FollowerTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class InferenceEngine:
    """Main inference engine: camera → detection → assessment → annotation."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.pose_detector = None
        self.yolo_model = None
        self.threat_model = None
        self.anomaly_model = None
        self.reid_model = None
        self.weapon_model = None

        self.threat_assessor = ThreatAssessor()
        self.follower_tracker = FollowerTracker()

        # State
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.running = False
        self.night_vision = True

        # Detection results cache
        self.last_pose = None
        self.last_objects = []
        self.current_threat = "NORMAL"
        self.confidence = 0
        self.detected_objects = []

        # Buffers
        self.pose_history = deque(maxlen=30)
        self.bbox_history = deque(maxlen=150)

        # COCO-SSD dangerous classes
        self.weapon_classes = {
            "baseball bat": ("CRITICAL", "BAT DETECTED"),
            "knife": ("CRITICAL", "KNIFE DETECTED"),
            "scissors": ("SUSPICIOUS", "SHARP OBJECT"),
        }
        self.suspicious_classes = {"bottle", "umbrella", "cell phone", "sports ball"}

    def initialize(self):
        """Load all models and start camera."""
        print("[SENTINEL] Initializing inference engine...")

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            model_complexity=settings.POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            min_detection_confidence=settings.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.POSE_MIN_TRACKING_CONFIDENCE,
        )
        print("[SENTINEL] MediaPipe Pose loaded")

        # YOLOv8 for object/weapon detection
        try:
            from ultralytics import YOLO
            yolo_path = Path(settings.MODELS_DIR) / settings.YOLO_MODEL
            if not yolo_path.exists():
                # Download default model
                self.yolo_model = YOLO("yolov8n.pt")
            else:
                self.yolo_model = YOLO(str(yolo_path))
            print("[SENTINEL] YOLOv8 loaded")
        except Exception as e:
            print(f"[SENTINEL] YOLOv8 load failed: {e}. Using MediaPipe only.")

        # Load trained ML models if available
        self._load_ml_models()

        # Camera
        self.cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[SENTINEL] Camera ready: {actual_w}x{actual_h}")

        self.running = True
        print("[SENTINEL] Inference engine initialized")

    def _load_ml_models(self):
        """Load trained PyTorch models if weight files exist."""
        models_dir = Path(settings.MODELS_DIR)

        # Threat classifier
        threat_path = models_dir / "threat_mlp.pt"
        if threat_path.exists():
            from sentinel.training.models import ThreatClassifierMLP
            self.threat_model = ThreatClassifierMLP().to(DEVICE)
            self.threat_model.load_state_dict(torch.load(threat_path, map_location=DEVICE, weights_only=True))
            self.threat_model.eval()
            print("[SENTINEL] Threat MLP model loaded")

        # Anomaly detector
        anomaly_path = models_dir / "anomaly_autoencoder.pt"
        if anomaly_path.exists():
            from sentinel.training.models import AnomalyAutoencoder
            checkpoint = torch.load(anomaly_path, map_location=DEVICE, weights_only=True)
            self.anomaly_model = AnomalyAutoencoder().to(DEVICE)
            self.anomaly_model.load_state_dict(checkpoint["model_state_dict"])
            self.anomaly_model.eval()
            self.anomaly_threshold = checkpoint.get("threshold", 0.01)
            print("[SENTINEL] Anomaly autoencoder loaded")

        # Re-ID net
        reid_path = models_dir / "reid_net.pt"
        if reid_path.exists():
            from sentinel.training.models import PersonReIDNet
            self.reid_model = PersonReIDNet().to(DEVICE)
            self.reid_model.load_state_dict(torch.load(reid_path, map_location=DEVICE, weights_only=True))
            self.reid_model.eval()
            print("[SENTINEL] Re-ID network loaded")

        # Weapon context
        weapon_path = models_dir / "weapon_context.pt"
        if weapon_path.exists():
            from sentinel.training.models import WeaponContextClassifier
            self.weapon_model = WeaponContextClassifier().to(DEVICE)
            self.weapon_model.load_state_dict(torch.load(weapon_path, map_location=DEVICE, weights_only=True))
            self.weapon_model.eval()
            print("[SENTINEL] Weapon context classifier loaded")

    def process_frame(self) -> Optional[dict]:
        """Capture and process a single frame. Returns detection results dict."""
        if not self.running or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_count += 1
        self._update_fps()

        h, w = frame.shape[:2]
        results = {
            "frame_number": self.frame_count,
            "fps": self.fps,
            "width": w,
            "height": h,
            "persons": [],
            "objects": [],
            "threat_level": "NORMAL",
            "confidence": 0,
            "weapon_detected": False,
            "weapon_label": "",
            "held_weapon": False,
            "anomaly_score": 0.0,
            "ml_threat_probs": None,
        }

        # --- Pose Detection ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose_detector.process(rgb_frame)

        person_detected = False
        landmarks = None
        person_bbox = None
        wrist_positions = []

        if pose_results.pose_landmarks:
            person_detected = True
            landmarks = pose_results.pose_landmarks.landmark

            # Extract bbox and wrists
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            min_x, max_x = min(xs) - 20, max(xs) + 20
            min_y, max_y = min(ys) - 20, max(ys) + 20
            person_bbox = {
                "x": max(0, min_x), "y": max(0, min_y),
                "w": min(w, max_x) - max(0, min_x),
                "h": min(h, max_y) - max(0, min_y),
            }

            # Wrist positions for held-weapon detection
            lw = landmarks[15]
            rw = landmarks[16]
            wrist_positions = [
                {"x": lw.x * w, "y": lw.y * h},
                {"x": rw.x * w, "y": rw.y * h},
            ]

            # Build landmark array for ML
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            self.pose_history.append(lm_array.flatten())

            results["persons"].append({
                "bbox": person_bbox,
                "landmarks": [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                "wrists": wrist_positions,
            })

        # --- Object Detection (YOLO, every 3 frames) ---
        if self.yolo_model and self.frame_count % 3 == 0:
            try:
                yolo_results = self.yolo_model(frame, conf=settings.YOLO_CONFIDENCE_THRESHOLD, verbose=False)
                for det in yolo_results[0].boxes:
                    cls_id = int(det.cls[0])
                    cls_name = self.yolo_model.names[cls_id]
                    conf = float(det.conf[0])
                    box = det.xyxy[0].cpu().numpy()
                    bx, by, bx2, by2 = box
                    bw, bh = bx2 - bx, by2 - by

                    if cls_name == "person":
                        if not person_detected:
                            person_detected = True
                            person_bbox = {"x": float(bx), "y": float(by), "w": float(bw), "h": float(bh)}
                        continue

                    obj_info = {
                        "label": cls_name,
                        "confidence": round(conf * 100),
                        "bbox": [float(bx), float(by), float(bw), float(bh)],
                        "level": "normal",
                        "held": False,
                    }

                    # Check weapon classes
                    if cls_name in self.weapon_classes:
                        level, label = self.weapon_classes[cls_name]
                        obj_info["level"] = "weapon"
                        obj_info["weapon_label"] = label
                        obj_info["held"] = self._is_held([bx, by, bw, bh], wrist_positions)
                        results["weapon_detected"] = True
                        results["weapon_label"] = label
                        if obj_info["held"]:
                            results["held_weapon"] = True
                    elif cls_name in self.suspicious_classes:
                        obj_info["level"] = "suspicious"

                    results["objects"].append(obj_info)
                    self.last_objects = results["objects"]
            except Exception as e:
                print(f"[YOLO ERROR] {e}")

        else:
            results["objects"] = self.last_objects

        # --- ML Model Inference ---
        if person_detected and landmarks:
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            pose_flat = lm_array.flatten()

            # Threat MLP
            if self.threat_model:
                features = self._build_pose_features(lm_array)
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                probs = self.threat_model.predict_proba(x)[0].cpu().numpy()
                results["ml_threat_probs"] = {
                    "normal": float(probs[0]),
                    "suspicious": float(probs[1]),
                    "critical": float(probs[2]),
                }

            # Anomaly detection
            if self.anomaly_model:
                x = torch.tensor(pose_flat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                score = float(self.anomaly_model.anomaly_score(x)[0].cpu())
                results["anomaly_score"] = score

            # Re-ID embedding for follower tracking
            if self.reid_model:
                x = torch.tensor(pose_flat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                embedding = self.reid_model.get_embedding(x)[0].cpu().numpy()
                results["reid_embedding"] = embedding.tolist()

        # --- Threat Assessment ---
        assessment = self.threat_assessor.assess(
            person_detected=person_detected,
            person_bbox=person_bbox,
            landmarks=landmarks,
            objects=results["objects"],
            weapon_detected=results["weapon_detected"],
            held_weapon=results["held_weapon"],
            ml_probs=results.get("ml_threat_probs"),
            anomaly_score=results.get("anomaly_score", 0),
            bbox_history=self.bbox_history,
            frame_size=(w, h),
        )
        results["threat_level"] = assessment["level"]
        results["confidence"] = assessment["confidence"]
        results["assessment_details"] = assessment

        self.current_threat = assessment["level"]
        self.confidence = assessment["confidence"]

        # --- Follower Tracking ---
        if person_detected and person_bbox:
            follower_result = self.follower_tracker.track(
                person_bbox=person_bbox,
                frame_size=(w, h),
                threat_level=self.current_threat,
                objects=[o["label"] for o in results["objects"] if o["level"] in ("weapon", "suspicious")],
                reid_embedding=results.get("reid_embedding"),
                weapon_detected=results["weapon_detected"],
                weapon_label=results.get("weapon_label", ""),
            )
            results["follower"] = follower_result

        # --- Annotate Frame ---
        annotated = self._annotate_frame(frame, results)
        if self.night_vision:
            annotated = self._apply_night_vision(annotated)

        results["annotated_frame"] = annotated

        return results

    def _build_pose_features(self, lm_array: np.ndarray) -> np.ndarray:
        """Build 103-dim feature vector from landmarks."""
        flat = lm_array.flatten()  # 99

        # Derived features
        xs = lm_array[:, 0]
        ys = lm_array[:, 1]
        bbox_cov = (max(xs) - min(xs)) * (max(ys) - min(ys))

        wrist_above = float(lm_array[15, 1] < lm_array[0, 1] and lm_array[16, 1] < lm_array[0, 1])

        l_dist = np.linalg.norm(lm_array[15, :2] - lm_array[11, :2])
        r_dist = np.linalg.norm(lm_array[16, :2] - lm_array[12, :2])
        arm_ext = max(l_dist, r_dist)

        center = 0.5
        left_dists = np.abs(lm_array[[11,13,15,23,25,27], 0] - center)
        right_dists = np.abs(lm_array[[12,14,16,24,26,28], 0] - center)
        symmetry = 1.0 - np.mean(np.abs(left_dists - right_dists))

        return np.concatenate([flat, [bbox_cov, wrist_above, arm_ext, symmetry]])

    def _is_held(self, bbox, wrist_positions, margin=30):
        if not wrist_positions:
            return False
        bx, by, bw, bh = bbox
        for wrist in wrist_positions:
            if (bx - margin <= wrist["x"] <= bx + bw + margin and
                by - margin <= wrist["y"] <= by + bh + margin):
                return True
        return False

    def _annotate_frame(self, frame, results):
        """Draw overlays: skeleton, bounding boxes, labels."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Threat colors
        color_map = {
            "NORMAL": (0, 255, 65),
            "SUSPICIOUS": (0, 204, 255),
            "CRITICAL": (32, 32, 255),
        }
        threat_color = color_map.get(results["threat_level"], (0, 255, 65))

        # Draw person
        for person in results["persons"]:
            bbox = person["bbox"]
            cv2.rectangle(annotated,
                          (int(bbox["x"]), int(bbox["y"])),
                          (int(bbox["x"] + bbox["w"]), int(bbox["y"] + bbox["h"])),
                          threat_color, 2)

            # Skeleton
            lms = person["landmarks"]
            connections = [
                (11,12),(11,13),(13,15),(12,14),(14,16),
                (11,23),(12,24),(23,24),(23,25),(24,26),
                (25,27),(26,28),(27,29),(28,30),(29,31),(30,32)
            ]
            for a, b in connections:
                if lms[a][3] > 0.5 and lms[b][3] > 0.5:
                    pt1 = (int(lms[a][0] * w), int(lms[a][1] * h))
                    pt2 = (int(lms[b][0] * w), int(lms[b][1] * h))
                    cv2.line(annotated, pt1, pt2, (0, 255, 65), 2)

            for lm in lms:
                if lm[3] > 0.5:
                    cv2.circle(annotated, (int(lm[0]*w), int(lm[1]*h)), 3, (0, 255, 65), -1)

        # Draw objects
        for obj in results["objects"]:
            bx, by, bw, bh = obj["bbox"]
            if obj["level"] == "weapon":
                color = (0, 0, 255)
                cv2.rectangle(annotated, (int(bx), int(by)),
                              (int(bx+bw), int(by+bh)), color, 2)
                label = f"{obj.get('weapon_label', obj['label'])} {obj['confidence']}%"
                cv2.putText(annotated, label, (int(bx), int(by)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if obj.get("held"):
                    cv2.putText(annotated, "HELD WEAPON", (int(bx), int(by+bh+18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif obj["level"] == "suspicious":
                color = (0, 204, 255)
                cv2.rectangle(annotated, (int(bx), int(by)),
                              (int(bx+bw), int(by+bh)), color, 1)
                cv2.putText(annotated, f"{obj['label']} {obj['confidence']}%",
                            (int(bx), int(by)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # HUD overlays
        cv2.putText(annotated, f"FPS: {self.fps}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 65), 1)
        cv2.putText(annotated, f"THREAT: {results['threat_level']} ({results['confidence']}%)",
                     (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, threat_color, 2)

        # Crosshair
        cx, cy = w // 2, h // 2
        cv2.line(annotated, (cx - 15, cy), (cx + 15, cy), (0, 255, 65), 1)
        cv2.line(annotated, (cx, cy - 15), (cx, cy + 15), (0, 255, 65), 1)

        # Corner brackets
        blen = 30
        bc = threat_color
        cv2.line(annotated, (15, 15), (15 + blen, 15), bc, 2)
        cv2.line(annotated, (15, 15), (15, 15 + blen), bc, 2)
        cv2.line(annotated, (w-15, 15), (w-15-blen, 15), bc, 2)
        cv2.line(annotated, (w-15, 15), (w-15, 15+blen), bc, 2)
        cv2.line(annotated, (15, h-15), (15+blen, h-15), bc, 2)
        cv2.line(annotated, (15, h-15), (15, h-15-blen), bc, 2)
        cv2.line(annotated, (w-15, h-15), (w-15-blen, h-15), bc, 2)
        cv2.line(annotated, (w-15, h-15), (w-15, h-15-blen), bc, 2)

        return annotated

    def _apply_night_vision(self, frame):
        """Green phosphor night vision filter."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        noise = np.random.normal(0, 8, gray.shape).astype(np.float32)

        nv = np.zeros_like(frame, dtype=np.float32)
        nv[:, :, 0] = np.clip((gray + noise) * 0.15, 0, 255)  # B
        nv[:, :, 1] = np.clip((gray + noise) * 1.1, 0, 255)   # G
        nv[:, :, 2] = np.clip((gray + noise) * 0.2, 0, 255)    # R

        result = nv.astype(np.uint8)

        # Scanlines
        h = result.shape[0]
        for row in range(0, h, 3):
            result[row, :] = (result[row, :].astype(np.float32) * 0.85).astype(np.uint8)

        # Vignette
        rows, cols = result.shape[:2]
        X = np.arange(cols) - cols / 2
        Y = np.arange(rows) - rows / 2
        X, Y = np.meshgrid(X, Y)
        dist = np.sqrt(X**2 + Y**2)
        max_dist = np.sqrt((cols/2)**2 + (rows/2)**2)
        vignette = 1.0 - np.clip((dist / max_dist - 0.4) * 1.5, 0, 0.7)
        for c in range(3):
            result[:, :, c] = (result[:, :, c].astype(np.float32) * vignette).astype(np.uint8)

        return result

    def _update_fps(self):
        self.fps_counter += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = now

    def get_jpeg_frame(self, results: dict) -> bytes:
        """Encode annotated frame as JPEG bytes."""
        frame = results.get("annotated_frame")
        if frame is None:
            return b""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()

    def release(self):
        """Clean up resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.pose_detector:
            self.pose_detector.close()
        print("[SENTINEL] Inference engine stopped")
