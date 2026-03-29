"""
Synthetic data generation for SENTINEL ML training.
Generates pose-based threat samples and augmented training data
when real labeled data is insufficient.
"""
import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from tqdm import tqdm

from sentinel.config import settings


# 33 MediaPipe pose landmarks
NUM_LANDMARKS = 33
LANDMARK_DIMS = 4  # x, y, z, visibility


class PoseSampleGenerator:
    """Generate synthetic pose landmark samples for threat classification training."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or settings.TRAINING_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_standing_pose(self) -> np.ndarray:
        """Generate a neutral standing pose (normalized 0-1 coordinates)."""
        landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIMS))
        # Key landmark positions for a standing person facing camera
        positions = {
            0: (0.50, 0.12),   # nose
            1: (0.49, 0.10),   # left eye inner
            2: (0.48, 0.10),   # left eye
            3: (0.47, 0.10),   # left eye outer
            4: (0.51, 0.10),   # right eye inner
            5: (0.52, 0.10),   # right eye
            6: (0.53, 0.10),   # right eye outer
            7: (0.46, 0.11),   # left ear
            8: (0.54, 0.11),   # right ear
            9: (0.49, 0.14),   # mouth left
            10: (0.51, 0.14),  # mouth right
            11: (0.42, 0.25),  # left shoulder
            12: (0.58, 0.25),  # right shoulder
            13: (0.38, 0.40),  # left elbow
            14: (0.62, 0.40),  # right elbow
            15: (0.36, 0.55),  # left wrist
            16: (0.64, 0.55),  # right wrist
            17: (0.35, 0.56),  # left pinky
            18: (0.65, 0.56),  # right pinky
            19: (0.34, 0.55),  # left index
            20: (0.66, 0.55),  # right index
            21: (0.36, 0.57),  # left thumb
            22: (0.64, 0.57),  # right thumb
            23: (0.44, 0.52),  # left hip
            24: (0.56, 0.52),  # right hip
            25: (0.43, 0.70),  # left knee
            26: (0.57, 0.70),  # right knee
            27: (0.42, 0.88),  # left ankle
            28: (0.58, 0.88),  # right ankle
            29: (0.41, 0.91),  # left heel
            30: (0.59, 0.91),  # right heel
            31: (0.40, 0.92),  # left foot index
            32: (0.60, 0.92),  # right foot index
        }
        for idx, (x, y) in positions.items():
            landmarks[idx] = [x, y, 0.0, 0.95]
        return landmarks

    def _add_noise(self, landmarks: np.ndarray, scale: float = 0.02) -> np.ndarray:
        noise = np.random.normal(0, scale, landmarks.shape)
        noise[:, 3] = 0  # don't add noise to visibility
        return np.clip(landmarks + noise, 0, 1)

    def generate_normal_pose(self) -> np.ndarray:
        """Normal: arms at sides, relaxed standing."""
        pose = self._base_standing_pose()
        # Slight random arm position variations
        pose[15][1] = random.uniform(0.50, 0.60)  # wrists near hip level
        pose[16][1] = random.uniform(0.50, 0.60)
        return self._add_noise(pose)

    def generate_suspicious_loitering(self) -> np.ndarray:
        """Suspicious: standing still, slight sway, arms may be crossed or in pockets."""
        pose = self._base_standing_pose()
        # Arms crossed or close to body
        pose[15][0] = random.uniform(0.45, 0.55)
        pose[16][0] = random.uniform(0.45, 0.55)
        pose[15][1] = random.uniform(0.35, 0.45)
        pose[16][1] = random.uniform(0.35, 0.45)
        return self._add_noise(pose, scale=0.015)

    def generate_suspicious_approach(self) -> np.ndarray:
        """Suspicious: person approaching (scaled up)."""
        pose = self._base_standing_pose()
        scale = random.uniform(1.1, 1.4)
        center_x, center_y = 0.5, 0.5
        for i in range(NUM_LANDMARKS):
            pose[i][0] = center_x + (pose[i][0] - center_x) * scale
            pose[i][1] = center_y + (pose[i][1] - center_y) * scale
        return self._add_noise(pose)

    def generate_aggressive_raised_arms(self) -> np.ndarray:
        """Critical: both wrists above nose level."""
        pose = self._base_standing_pose()
        nose_y = pose[0][1]
        pose[13][1] = random.uniform(0.08, 0.15)  # elbows high
        pose[14][1] = random.uniform(0.08, 0.15)
        pose[15][0] = random.uniform(0.35, 0.50)   # wrists above head
        pose[15][1] = random.uniform(0.02, nose_y - 0.02)
        pose[16][0] = random.uniform(0.50, 0.65)
        pose[16][1] = random.uniform(0.02, nose_y - 0.02)
        return self._add_noise(pose)

    def generate_running_toward(self) -> np.ndarray:
        """Critical: running pose, large bounding box."""
        pose = self._base_standing_pose()
        # Running: one leg forward, arms pumping
        pose[25][1] = random.uniform(0.60, 0.68)  # left knee forward
        pose[26][1] = random.uniform(0.72, 0.80)  # right knee back
        pose[13][1] = random.uniform(0.28, 0.35)  # arms pumping
        pose[14][1] = random.uniform(0.38, 0.45)
        pose[15][1] = random.uniform(0.30, 0.40)
        pose[16][1] = random.uniform(0.42, 0.50)
        # Scale up = close proximity
        scale = random.uniform(1.3, 1.8)
        center_x, center_y = 0.5, 0.5
        for i in range(NUM_LANDMARKS):
            pose[i][0] = center_x + (pose[i][0] - center_x) * scale
            pose[i][1] = center_y + (pose[i][1] - center_y) * scale
        return self._add_noise(pose, scale=0.025)

    def generate_weapon_holding(self) -> np.ndarray:
        """Critical: arm extended holding something (wrist far from body)."""
        pose = self._base_standing_pose()
        # One arm extended forward/outward
        side = random.choice(["left", "right"])
        if side == "left":
            pose[13][0] = random.uniform(0.25, 0.35)
            pose[13][1] = random.uniform(0.25, 0.35)
            pose[15][0] = random.uniform(0.15, 0.30)
            pose[15][1] = random.uniform(0.20, 0.35)
        else:
            pose[14][0] = random.uniform(0.65, 0.75)
            pose[14][1] = random.uniform(0.25, 0.35)
            pose[16][0] = random.uniform(0.70, 0.85)
            pose[16][1] = random.uniform(0.20, 0.35)
        return self._add_noise(pose)

    def generate_erratic_movement(self) -> np.ndarray:
        """Suspicious: unusual joint angles, erratic posture."""
        pose = self._base_standing_pose()
        # Random large perturbations to several joints
        for idx in [13, 14, 15, 16, 25, 26]:
            pose[idx][0] += random.uniform(-0.15, 0.15)
            pose[idx][1] += random.uniform(-0.15, 0.15)
        return self._add_noise(pose, scale=0.03)

    def generate_dataset(self, samples_per_class: int = 1000) -> dict:
        """Generate full training dataset with labels."""
        data = {"poses": [], "labels": [], "metadata": []}

        generators = {
            "normal": [
                (self.generate_normal_pose, 1.0),
            ],
            "suspicious": [
                (self.generate_suspicious_loitering, 0.4),
                (self.generate_suspicious_approach, 0.3),
                (self.generate_erratic_movement, 0.3),
            ],
            "critical": [
                (self.generate_aggressive_raised_arms, 0.3),
                (self.generate_running_toward, 0.3),
                (self.generate_weapon_holding, 0.4),
            ],
        }

        label_map = {"normal": 0, "suspicious": 1, "critical": 2}

        for label, gens in generators.items():
            for i in tqdm(range(samples_per_class), desc=f"Generating {label}"):
                # Weighted random pick
                weights = [w for _, w in gens]
                gen_fn = random.choices([g for g, _ in gens], weights=weights, k=1)[0]
                pose = gen_fn()

                # Flatten to feature vector
                features = pose[:, :3].flatten()  # x, y, z for 33 landmarks = 99 features

                # Add derived features
                bbox_coverage = self._compute_bbox_coverage(pose)
                wrist_above_nose = float(
                    pose[15][1] < pose[0][1] and pose[16][1] < pose[0][1]
                )
                arm_extension = self._compute_arm_extension(pose)
                body_symmetry = self._compute_symmetry(pose)

                extra_features = np.array([
                    bbox_coverage, wrist_above_nose, arm_extension, body_symmetry
                ])
                full_features = np.concatenate([features, extra_features])

                data["poses"].append(full_features.tolist())
                data["labels"].append(label_map[label])
                data["metadata"].append({
                    "label_name": label,
                    "generator": gen_fn.__name__,
                })

        return data

    def _compute_bbox_coverage(self, pose: np.ndarray) -> float:
        xs = pose[:, 0]
        ys = pose[:, 1]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        return float(w * h)

    def _compute_arm_extension(self, pose: np.ndarray) -> float:
        l_shoulder = pose[11][:2]
        l_wrist = pose[15][:2]
        r_shoulder = pose[12][:2]
        r_wrist = pose[16][:2]
        l_dist = np.linalg.norm(l_wrist - l_shoulder)
        r_dist = np.linalg.norm(r_wrist - r_shoulder)
        return float(max(l_dist, r_dist))

    def _compute_symmetry(self, pose: np.ndarray) -> float:
        center_x = 0.5
        left_pts = pose[[11, 13, 15, 23, 25, 27], 0]
        right_pts = pose[[12, 14, 16, 24, 26, 28], 0]
        left_dists = np.abs(left_pts - center_x)
        right_dists = np.abs(right_pts - center_x)
        return float(1.0 - np.mean(np.abs(left_dists - right_dists)))

    def save_dataset(self, data: dict, filename: str = "threat_training_data.json"):
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved {len(data['labels'])} samples to {path}")
        return path


class FrameAugmentor:
    """Image augmentation pipeline for weapon detection fine-tuning."""

    @staticmethod
    def augment_frame(frame: np.ndarray) -> list[np.ndarray]:
        """Apply augmentations to a single frame, return list of augmented frames."""
        augmented = [frame]

        # Horizontal flip
        augmented.append(cv2.flip(frame, 1))

        # Brightness variations
        for beta in [-30, -15, 15, 30]:
            augmented.append(cv2.convertScaleAbs(frame, alpha=1.0, beta=beta))

        # Contrast variations
        for alpha in [0.7, 0.85, 1.15, 1.3]:
            augmented.append(cv2.convertScaleAbs(frame, alpha=alpha, beta=0))

        # Gaussian blur (simulates camera defocus)
        augmented.append(cv2.GaussianBlur(frame, (5, 5), 0))

        # Add noise
        noise = np.random.normal(0, 15, frame.shape).astype(np.uint8)
        augmented.append(cv2.add(frame, noise))

        # Night vision simulation (green channel emphasis)
        nv = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nv[:, :, 0] = np.clip(gray * 0.15, 0, 255).astype(np.uint8)
        nv[:, :, 1] = np.clip(gray * 1.1, 0, 255).astype(np.uint8)
        nv[:, :, 2] = np.clip(gray * 0.2, 0, 255).astype(np.uint8)
        augmented.append(nv)

        return augmented

    @staticmethod
    def create_mosaic(frames: list[np.ndarray], grid_size: int = 2) -> np.ndarray:
        """Create mosaic augmentation from multiple frames."""
        h, w = frames[0].shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        mosaic = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(grid_size):
            for j in range(grid_size):
                idx = (i * grid_size + j) % len(frames)
                cell = cv2.resize(frames[idx], (cell_w, cell_h))
                mosaic[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = cell
        return mosaic


class SequenceDataGenerator:
    """Generate temporal sequence data for LSTM-based threat prediction."""

    def __init__(self, sequence_length: int = 30):
        self.seq_len = sequence_length
        self.pose_gen = PoseSampleGenerator()

    def generate_normal_sequence(self) -> list[np.ndarray]:
        """Stationary normal sequence with minor drift."""
        base = self.pose_gen.generate_normal_pose()
        sequence = []
        for t in range(self.seq_len):
            frame = base.copy()
            frame = self.pose_gen._add_noise(frame, scale=0.005)
            sequence.append(frame[:, :3].flatten())
        return sequence

    def generate_approach_sequence(self) -> list[np.ndarray]:
        """Person gradually getting closer (bbox growing)."""
        base = self.pose_gen._base_standing_pose()
        sequence = []
        for t in range(self.seq_len):
            scale = 1.0 + (t / self.seq_len) * 0.8  # 1.0 to 1.8
            frame = base.copy()
            cx, cy = 0.5, 0.5
            for i in range(NUM_LANDMARKS):
                frame[i][0] = cx + (frame[i][0] - cx) * scale
                frame[i][1] = cy + (frame[i][1] - cy) * scale
            frame = self.pose_gen._add_noise(frame, scale=0.01)
            sequence.append(frame[:, :3].flatten())
        return sequence

    def generate_sudden_aggression_sequence(self) -> list[np.ndarray]:
        """Normal then suddenly raises arms."""
        sequence = []
        transition_point = random.randint(15, 25)
        for t in range(self.seq_len):
            if t < transition_point:
                pose = self.pose_gen.generate_normal_pose()
            else:
                progress = (t - transition_point) / (self.seq_len - transition_point)
                pose = self.pose_gen._base_standing_pose()
                # Gradually raise arms
                pose[15][1] = 0.55 - progress * 0.50
                pose[16][1] = 0.55 - progress * 0.50
                pose[13][1] = 0.40 - progress * 0.30
                pose[14][1] = 0.40 - progress * 0.30
                pose = self.pose_gen._add_noise(pose)
            sequence.append(pose[:, :3].flatten())
        return sequence

    def generate_dataset(self, sequences_per_class: int = 500) -> dict:
        """Generate temporal sequence dataset."""
        data = {"sequences": [], "labels": []}

        generators = {
            0: self.generate_normal_sequence,
            1: self.generate_approach_sequence,
            2: self.generate_sudden_aggression_sequence,
        }
        label_names = {0: "normal", 1: "suspicious", 2: "critical"}

        for label, gen_fn in generators.items():
            for _ in tqdm(range(sequences_per_class), desc=f"Sequences: {label_names[label]}"):
                seq = gen_fn()
                data["sequences"].append([s.tolist() for s in seq])
                data["labels"].append(label)

        return data


if __name__ == "__main__":
    print("=== Generating Pose Threat Training Data ===")
    gen = PoseSampleGenerator()
    data = gen.generate_dataset(samples_per_class=2000)
    gen.save_dataset(data)

    print("\n=== Generating Temporal Sequence Data ===")
    seq_gen = SequenceDataGenerator(sequence_length=30)
    seq_data = seq_gen.generate_dataset(sequences_per_class=500)
    path = gen.output_dir / "sequence_training_data.json"
    with open(path, "w") as f:
        json.dump(seq_data, f)
    print(f"Saved {len(seq_data['labels'])} sequences to {path}")
