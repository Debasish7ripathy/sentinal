"""
Visualization utilities for SENTINEL training and evaluation.
Generates plots for model performance, threat distributions, and follower analytics.
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history: dict, model_name: str, output_dir: str = "models/plots"):
    """Plot training/validation loss and accuracy curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"SENTINEL — {model_name} Training", fontsize=14, color='#00ff41')
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#00aa2a')
        ax.xaxis.label.set_color('#00aa2a')
        ax.yaxis.label.set_color('#00aa2a')
        ax.spines['bottom'].set_color('#0a3a0a')
        ax.spines['left'].set_color('#0a3a0a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Loss
    axes[0].plot(history["train_loss"], color='#00ff41', label='Train Loss', linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], color='#ffcc00', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss', color='#00ff41')
    axes[0].legend(facecolor='#0a0a0a', edgecolor='#0a3a0a', labelcolor='#00ff41')

    # Accuracy
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], color='#00ff41', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy', color='#00ff41')

    plt.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_training.png")
    plt.savefig(path, facecolor='#0a0a0a', dpi=150)
    plt.close()
    return path


def plot_confusion_matrix(cm: np.ndarray, labels: list, model_name: str, output_dir: str = "models/plots"):
    """Plot confusion matrix with military-HUD aesthetic."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=1, linecolor='#0a3a0a',
        cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel('Predicted', color='#00aa2a')
    ax.set_ylabel('Actual', color='#00aa2a')
    ax.set_title(f'{model_name} — Confusion Matrix', color='#00ff41')
    ax.tick_params(colors='#00aa2a')

    path = os.path.join(output_dir, f"{model_name}_confusion.png")
    plt.tight_layout()
    plt.savefig(path, facecolor='#0a0a0a', dpi=150)
    plt.close()
    return path


def plot_anomaly_distribution(normal_scores, suspicious_scores, critical_scores,
                               threshold: float, output_dir: str = "models/plots"):
    """Plot anomaly score distributions per class."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    ax.hist(normal_scores, bins=50, alpha=0.7, color='#00ff41', label='Normal')
    ax.hist(suspicious_scores, bins=50, alpha=0.7, color='#ffcc00', label='Suspicious')
    ax.hist(critical_scores, bins=50, alpha=0.7, color='#ff2020', label='Critical')
    ax.axvline(x=threshold, color='#ff7700', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')

    ax.set_xlabel('Anomaly Score', color='#00aa2a')
    ax.set_ylabel('Count', color='#00aa2a')
    ax.set_title('Anomaly Score Distribution', color='#00ff41')
    ax.legend(facecolor='#0a0a0a', edgecolor='#0a3a0a', labelcolor='#00ff41')
    ax.tick_params(colors='#00aa2a')
    ax.spines['bottom'].set_color('#0a3a0a')
    ax.spines['left'].set_color('#0a3a0a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(output_dir, "anomaly_distribution.png")
    plt.tight_layout()
    plt.savefig(path, facecolor='#0a0a0a', dpi=150)
    plt.close()
    return path
