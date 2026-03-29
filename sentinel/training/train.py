"""
SENTINEL ML Training Pipeline.

Trains all models:
1. ThreatClassifierMLP — pose-based threat classification
2. ThreatSequenceLSTM — temporal threat prediction
3. AnomalyAutoencoder — unsupervised anomaly detection
4. PersonReIDNet — follower re-identification
5. WeaponContextClassifier — weapon + pose fusion

Usage:
    python -m sentinel.training.train --all
    python -m sentinel.training.train --model threat_mlp
    python -m sentinel.training.train --model lstm
    python -m sentinel.training.train --model anomaly
    python -m sentinel.training.train --model reid
    python -m sentinel.training.train --model weapon
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from sentinel.config import settings
from sentinel.training.models import (
    ThreatClassifierMLP,
    ThreatSequenceLSTM,
    AnomalyAutoencoder,
    PersonReIDNet,
    TripletLoss,
    WeaponContextClassifier,
    count_parameters,
)
from sentinel.training.data_generator import PoseSampleGenerator, SequenceDataGenerator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODELS_DIR = Path(settings.MODELS_DIR)
DATA_DIR = Path(settings.TRAINING_DATA_DIR)
LABEL_NAMES = ["normal", "suspicious", "critical"]


def load_or_generate_pose_data(samples_per_class: int = 2000):
    """Load existing data or generate synthetic data."""
    data_path = DATA_DIR / "threat_training_data.json"
    if data_path.exists():
        print(f"Loading existing data from {data_path}")
        with open(data_path) as f:
            data = json.load(f)
    else:
        print("Generating synthetic training data...")
        gen = PoseSampleGenerator()
        data = gen.generate_dataset(samples_per_class=samples_per_class)
        gen.save_dataset(data)
    return data


def load_or_generate_sequence_data(sequences_per_class: int = 500):
    data_path = DATA_DIR / "sequence_training_data.json"
    if data_path.exists():
        print(f"Loading existing sequence data from {data_path}")
        with open(data_path) as f:
            data = json.load(f)
    else:
        print("Generating synthetic sequence data...")
        gen = SequenceDataGenerator(sequence_length=30)
        data = gen.generate_dataset(sequences_per_class=sequences_per_class)
        with open(data_path, "w") as f:
            json.dump(data, f)
    return data


# ============================================================
# Training: ThreatClassifierMLP
# ============================================================
def train_threat_mlp(epochs: int = None, batch_size: int = None, lr: float = None):
    epochs = epochs or settings.TRAIN_EPOCHS
    batch_size = batch_size or settings.TRAIN_BATCH_SIZE
    lr = lr or settings.TRAIN_LR

    print("\n" + "=" * 60)
    print("TRAINING: ThreatClassifierMLP")
    print("=" * 60)

    data = load_or_generate_pose_data()
    X = torch.tensor(data["poses"], dtype=torch.float32)
    y = torch.tensor(data["labels"], dtype=torch.long)

    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * settings.TRAIN_VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ThreatClassifierMLP(input_dim=X.shape[1]).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Device: {DEVICE}")
    print(f"Train: {train_size}, Val: {val_size}")

    # Class weights for imbalanced data
    class_counts = torch.bincount(y)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "threat_mlp.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"Model saved: {MODELS_DIR / 'threat_mlp.pt'}")

    return model, history


# ============================================================
# Training: ThreatSequenceLSTM
# ============================================================
def train_threat_lstm(epochs: int = None, batch_size: int = None, lr: float = None):
    epochs = epochs or settings.TRAIN_EPOCHS
    batch_size = batch_size or settings.TRAIN_BATCH_SIZE
    lr = lr or settings.TRAIN_LR

    print("\n" + "=" * 60)
    print("TRAINING: ThreatSequenceLSTM")
    print("=" * 60)

    data = load_or_generate_sequence_data()
    X = torch.tensor(data["sequences"], dtype=torch.float32)
    y = torch.tensor(data["labels"], dtype=torch.long)

    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * settings.TRAIN_VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ThreatSequenceLSTM(input_dim=X.shape[2]).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Input shape: {X.shape} (batch, seq_len, features)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "threat_lstm.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))
    print(f"Model saved: {MODELS_DIR / 'threat_lstm.pt'}")

    return model


# ============================================================
# Training: AnomalyAutoencoder
# ============================================================
def train_anomaly_autoencoder(epochs: int = None, batch_size: int = None, lr: float = None):
    epochs = epochs or settings.TRAIN_EPOCHS
    batch_size = batch_size or settings.TRAIN_BATCH_SIZE
    lr = lr or settings.TRAIN_LR

    print("\n" + "=" * 60)
    print("TRAINING: AnomalyAutoencoder (normal poses only)")
    print("=" * 60)

    data = load_or_generate_pose_data()
    X = np.array(data["poses"])
    y = np.array(data["labels"])

    # Train on NORMAL only (label=0), test on all
    normal_mask = y == 0
    X_normal = torch.tensor(X[normal_mask][:, :99], dtype=torch.float32)  # Only pose coords
    X_all = torch.tensor(X[:, :99], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_normal), batch_size=batch_size, shuffle=True)

    model = AnomalyAutoencoder(input_dim=99).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Training on {len(X_normal)} normal samples")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            reconstruction = model(batch_x)
            loss = nn.MSELoss()(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Recon Loss: {total_loss/len(train_loader):.6f}")

    # Evaluate anomaly detection
    model.eval()
    with torch.no_grad():
        scores = model.anomaly_score(X_all.to(DEVICE)).cpu().numpy()

    normal_scores = scores[y == 0]
    suspicious_scores = scores[y == 1]
    critical_scores = scores[y == 2]

    print(f"\nAnomaly Scores (mean ± std):")
    print(f"  Normal:     {normal_scores.mean():.6f} ± {normal_scores.std():.6f}")
    print(f"  Suspicious: {suspicious_scores.mean():.6f} ± {suspicious_scores.std():.6f}")
    print(f"  Critical:   {critical_scores.mean():.6f} ± {critical_scores.std():.6f}")

    # Save threshold as 95th percentile of normal scores
    threshold = float(np.percentile(normal_scores, 95))
    print(f"  Anomaly threshold (95th %ile of normal): {threshold:.6f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
    }, MODELS_DIR / "anomaly_autoencoder.pt")
    print(f"Model saved: {MODELS_DIR / 'anomaly_autoencoder.pt'}")

    return model, threshold


# ============================================================
# Training: PersonReIDNet
# ============================================================
def train_reid_net(epochs: int = None, batch_size: int = None, lr: float = None):
    epochs = epochs or 30
    batch_size = batch_size or settings.TRAIN_BATCH_SIZE
    lr = lr or 0.0005

    print("\n" + "=" * 60)
    print("TRAINING: PersonReIDNet (Triplet Loss)")
    print("=" * 60)

    # Generate synthetic identity data
    gen = PoseSampleGenerator()
    num_identities = 50
    samples_per_identity = 20

    identities = {}
    for i in range(num_identities):
        base_pose = gen._base_standing_pose()
        # Each identity has unique body proportions
        scale_x = np.random.uniform(0.85, 1.15)
        scale_y = np.random.uniform(0.9, 1.1)
        base_pose[:, 0] = 0.5 + (base_pose[:, 0] - 0.5) * scale_x
        base_pose[:, 1] = 0.5 + (base_pose[:, 1] - 0.5) * scale_y

        samples = []
        for _ in range(samples_per_identity):
            pose = gen._add_noise(base_pose.copy(), scale=0.015)
            samples.append(torch.tensor(pose[:, :3].flatten(), dtype=torch.float32))
        identities[i] = torch.stack(samples)

    model = PersonReIDNet(input_dim=99, embedding_dim=64).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Identities: {num_identities}, Samples per ID: {samples_per_identity}")

    criterion = TripletLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_triplets = 0

        # Mine triplets in batches (BatchNorm needs batch_size > 1)
        anchors_batch, positives_batch, negatives_batch = [], [], []

        for anchor_id in range(num_identities):
            anchor_samples = identities[anchor_id]
            neg_ids = [j for j in range(num_identities) if j != anchor_id]

            for i in range(0, len(anchor_samples) - 1, 2):
                anchors_batch.append(anchor_samples[i])
                positives_batch.append(anchor_samples[i + 1])

                neg_id = np.random.choice(neg_ids)
                neg_idx = np.random.randint(len(identities[neg_id]))
                negatives_batch.append(identities[neg_id][neg_idx])

                if len(anchors_batch) >= batch_size:
                    a = torch.stack(anchors_batch).to(DEVICE)
                    p = torch.stack(positives_batch).to(DEVICE)
                    n = torch.stack(negatives_batch).to(DEVICE)

                    loss = criterion(model(a), model(p), model(n))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_triplets += 1
                    anchors_batch, positives_batch, negatives_batch = [], [], []

        # Process remaining triplets
        if len(anchors_batch) > 1:
            a = torch.stack(anchors_batch).to(DEVICE)
            p = torch.stack(positives_batch).to(DEVICE)
            n = torch.stack(negatives_batch).to(DEVICE)
            loss = criterion(model(a), model(p), model(n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_triplets += 1

        avg_loss = total_loss / max(num_triplets, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Triplet Loss: {avg_loss:.6f}")

    # Evaluate: check embedding separation
    model.eval()
    with torch.no_grad():
        intra_dists = []
        inter_dists = []
        for i in range(min(10, num_identities)):
            embs_i = model(identities[i].to(DEVICE))
            # Intra-class
            for a in range(len(embs_i)):
                for b in range(a + 1, len(embs_i)):
                    intra_dists.append(torch.sum((embs_i[a] - embs_i[b]) ** 2).item())

            # Inter-class
            for j in range(i + 1, min(10, num_identities)):
                embs_j = model(identities[j].to(DEVICE))
                inter_dists.append(torch.sum((embs_i[0] - embs_j[0]) ** 2).item())

    print(f"\nEmbedding Separation:")
    print(f"  Intra-class distance (same person): {np.mean(intra_dists):.4f} ± {np.std(intra_dists):.4f}")
    print(f"  Inter-class distance (diff person): {np.mean(inter_dists):.4f} ± {np.std(inter_dists):.4f}")
    print(f"  Ratio (higher=better): {np.mean(inter_dists)/max(np.mean(intra_dists), 1e-6):.2f}")

    torch.save(model.state_dict(), MODELS_DIR / "reid_net.pt")
    print(f"Model saved: {MODELS_DIR / 'reid_net.pt'}")

    return model


# ============================================================
# Training: WeaponContextClassifier
# ============================================================
def train_weapon_context(epochs: int = None, batch_size: int = None, lr: float = None):
    epochs = epochs or settings.TRAIN_EPOCHS
    batch_size = batch_size or settings.TRAIN_BATCH_SIZE
    lr = lr or settings.TRAIN_LR

    print("\n" + "=" * 60)
    print("TRAINING: WeaponContextClassifier")
    print("=" * 60)

    # Generate combined pose + object feature data
    pose_gen = PoseSampleGenerator()
    num_samples = 3000
    pose_features = []
    object_features = []
    labels = []

    for _ in tqdm(range(num_samples), desc="Generating weapon context data"):
        scenario = np.random.choice(["normal", "suspicious", "critical"], p=[0.4, 0.3, 0.3])

        if scenario == "normal":
            pose = pose_gen.generate_normal_pose()
            obj_feat = np.zeros(10)  # no weapon
            obj_feat[0] = 0  # weapon_detected flag
            labels.append(0)
        elif scenario == "suspicious":
            pose = np.random.choice([
                pose_gen.generate_suspicious_loitering,
                pose_gen.generate_suspicious_approach
            ])()
            obj_feat = np.zeros(10)
            # Sometimes has suspicious object
            if np.random.random() > 0.5:
                obj_feat[0] = 0  # not a weapon
                obj_feat[1] = np.random.uniform(0.3, 0.8)  # object confidence
                obj_feat[2] = np.random.uniform(0, 1)  # object bbox x
                obj_feat[3] = np.random.uniform(0, 1)  # object bbox y
                obj_feat[4] = np.random.uniform(0.05, 0.2)  # object size
                obj_feat[5] = 0  # is_held
            labels.append(1)
        else:  # critical
            pose = np.random.choice([
                pose_gen.generate_aggressive_raised_arms,
                pose_gen.generate_weapon_holding,
                pose_gen.generate_running_toward
            ])()
            obj_feat = np.zeros(10)
            obj_feat[0] = 1  # weapon detected
            obj_feat[1] = np.random.uniform(0.5, 0.99)  # confidence
            obj_feat[2] = np.random.uniform(0.2, 0.8)
            obj_feat[3] = np.random.uniform(0.2, 0.8)
            obj_feat[4] = np.random.uniform(0.03, 0.15)
            obj_feat[5] = float(np.random.random() > 0.3)  # is_held
            obj_feat[6] = np.random.choice([0, 1, 2])  # weapon_type (bat/knife/other)
            obj_feat[7] = np.random.uniform(0, 0.5)  # distance to person
            obj_feat[8] = np.random.uniform(0, 1)  # person velocity
            obj_feat[9] = np.random.uniform(0, 1)  # person bbox coverage
            labels.append(2)

        # Build pose feature vector (103 dims)
        flat_pose = pose[:, :3].flatten()  # 99
        bbox_cov = pose_gen._compute_bbox_coverage(pose)
        wrist_above = float(pose[15][1] < pose[0][1] and pose[16][1] < pose[0][1])
        arm_ext = pose_gen._compute_arm_extension(pose)
        symmetry = pose_gen._compute_symmetry(pose)
        full_pose = np.concatenate([flat_pose, [bbox_cov, wrist_above, arm_ext, symmetry]])

        pose_features.append(full_pose)
        object_features.append(obj_feat)

    X_pose = torch.tensor(np.array(pose_features), dtype=torch.float32)
    X_obj = torch.tensor(np.array(object_features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X_pose, X_obj, y)
    val_size = int(len(dataset) * 0.2)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = WeaponContextClassifier().to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_pose, batch_obj, batch_y in train_loader:
            batch_pose = batch_pose.to(DEVICE)
            batch_obj = batch_obj.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_pose, batch_obj)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_pose, batch_obj, batch_y in val_loader:
                outputs = model(batch_pose.to(DEVICE), batch_obj.to(DEVICE))
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y.to(DEVICE)).sum().item()

        val_acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "weapon_context.pt")

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Model saved: {MODELS_DIR / 'weapon_context.pt'}")

    return model


# ============================================================
# Main Entry
# ============================================================
def train_all():
    """Train all models sequentially."""
    start = time.time()
    print("=" * 60)
    print("  SENTINEL — FULL ML TRAINING PIPELINE")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    train_threat_mlp()
    train_threat_lstm()
    train_anomaly_autoencoder()
    train_reid_net()
    train_weapon_context()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  ALL MODELS TRAINED in {elapsed:.1f}s")
    print(f"  Weights saved to: {MODELS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL ML Training")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "threat_mlp", "lstm", "anomaly", "reid", "weapon"],
                        help="Which model to train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    if args.model == "all":
        train_all()
    elif args.model == "threat_mlp":
        train_threat_mlp(args.epochs, args.batch_size, args.lr)
    elif args.model == "lstm":
        train_threat_lstm(args.epochs, args.batch_size, args.lr)
    elif args.model == "anomaly":
        train_anomaly_autoencoder(args.epochs, args.batch_size, args.lr)
    elif args.model == "reid":
        train_reid_net(args.epochs, args.batch_size, args.lr)
    elif args.model == "weapon":
        train_weapon_context(args.epochs, args.batch_size, args.lr)
