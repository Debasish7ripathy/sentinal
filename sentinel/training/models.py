"""
PyTorch model architectures for SENTINEL ML pipeline.

Models:
1. ThreatClassifierMLP — Pose-based threat classification (Normal/Suspicious/Critical)
2. ThreatSequenceLSTM — Temporal threat prediction from pose sequences
3. PersonReIDNet — Person re-identification embedding network for follower tracking
4. AnomalyAutoencoder — Unsupervised anomaly detection on pose data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Threat Classifier MLP
# ============================================================
class ThreatClassifierMLP(nn.Module):
    """
    Multi-layer perceptron for classifying threat level from a single frame's
    pose landmarks + derived features.

    Input: 103-dim vector (33 landmarks * 3 coords + 4 derived features)
    Output: 3 classes (normal=0, suspicious=1, critical=2)
    """

    def __init__(self, input_dim: int = 103, hidden_dims: list = None, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


# ============================================================
# 2. Threat Sequence LSTM
# ============================================================
class ThreatSequenceLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal threat prediction.
    Processes a sequence of pose frames to predict threat escalation.

    Input: (batch, seq_len, 99) — 30 frames of 33*3 landmark coords
    Output: 3 classes
    """

    def __init__(self, input_dim: int = 99, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)

        return self.classifier(context)

    def predict_with_attention(self, x):
        self.eval()
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attn_weights = self.attention(lstm_out)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.sum(lstm_out * attn_weights, dim=1)
            logits = self.classifier(context)
            probs = F.softmax(logits, dim=-1)
            return probs, attn_weights.squeeze(-1)


# ============================================================
# 3. Person Re-ID Network
# ============================================================
class PersonReIDNet(nn.Module):
    """
    Lightweight person re-identification network.
    Generates embedding vectors from pose landmarks for matching
    the same person across sessions (privacy-safe, no face features).

    Input: 99-dim (33 landmarks * 3 coords)
    Output: 64-dim embedding vector
    """

    def __init__(self, input_dim: int = 99, embedding_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        # L2 normalize for cosine similarity
        return F.normalize(embedding, p=2, dim=-1)

    def get_embedding(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class TripletLoss(nn.Module):
    """Triplet loss for Re-ID training."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=-1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=-1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# ============================================================
# 4. Anomaly Autoencoder
# ============================================================
class AnomalyAutoencoder(nn.Module):
    """
    Autoencoder trained on NORMAL poses only.
    High reconstruction error = anomalous/threatening pose.

    Input: 99-dim pose
    Output: 99-dim reconstructed pose
    Anomaly score: MSE between input and reconstruction
    """

    def __init__(self, input_dim: int = 99, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)
            mse = torch.mean((x - reconstruction) ** 2, dim=-1)
            return mse


# ============================================================
# 5. Weapon Context Classifier
# ============================================================
class WeaponContextClassifier(nn.Module):
    """
    Classifies weapon threat context by combining:
    - Pose features (is person in aggressive posture?)
    - Object detection features (weapon type, confidence, is held?)
    - Spatial features (distance between weapon bbox and person)

    Input: pose_features(103) + object_features(10) = 113
    Output: threat_level (3 classes)
    """

    def __init__(self, pose_dim: int = 103, object_dim: int = 10, num_classes: int = 3):
        super().__init__()
        combined = pose_dim + object_dim

        self.pose_branch = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(inplace=True),
        )
        self.object_branch = nn.Sequential(
            nn.Linear(object_dim, 32),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, pose_features, object_features):
        p = self.pose_branch(pose_features)
        o = self.object_branch(object_features)
        combined = torch.cat([p, o], dim=-1)
        return self.fusion(combined)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== SENTINEL Model Architectures ===\n")

    models = {
        "ThreatClassifierMLP": ThreatClassifierMLP(),
        "ThreatSequenceLSTM": ThreatSequenceLSTM(),
        "PersonReIDNet": PersonReIDNet(),
        "AnomalyAutoencoder": AnomalyAutoencoder(),
        "WeaponContextClassifier": WeaponContextClassifier(),
    }

    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name}: {params:,} trainable parameters")
        print(f"  {model}\n")
