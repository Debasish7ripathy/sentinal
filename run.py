#!/usr/bin/env python3
"""
SENTINEL — Main Entry Point

Usage:
    python run.py serve          # Start the backend server + dashboard
    python run.py train          # Train all ML models
    python run.py train --model threat_mlp   # Train specific model
    python run.py generate-data  # Generate synthetic training data
    python run.py status         # Show system status
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def cli():
    """SENTINEL — Autonomous AI Personal Safety System v2.1"""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8099, type=int, help="Server port")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes")
def serve(host, port, reload):
    """Start the SENTINEL backend server and web dashboard."""
    console.print(Panel.fit(
        "[bold green]SENTINEL AUTONOMOUS SAFETY SYSTEM v2.1[/bold green]\n"
        f"[dim]Starting server on {host}:{port}[/dim]",
        border_style="green",
    ))

    import uvicorn
    from sentinel.models.database import init_db
    init_db()

    console.print(f"[green]Dashboard:[/green] http://localhost:{port}")
    console.print(f"[green]API Docs:[/green]  http://localhost:{port}/docs")
    console.print(f"[green]WebSocket:[/green] ws://localhost:{port}/ws/stream")
    console.print()

    uvicorn.run(
        "sentinel.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@cli.command()
@click.option("--model", default="all",
              type=click.Choice(["all", "threat_mlp", "lstm", "anomaly", "reid", "weapon"]),
              help="Which model to train")
@click.option("--epochs", default=None, type=int, help="Override epochs")
@click.option("--batch-size", default=None, type=int, help="Override batch size")
@click.option("--lr", default=None, type=float, help="Override learning rate")
def train(model, epochs, batch_size, lr):
    """Train ML models for threat detection."""
    console.print(Panel.fit(
        f"[bold green]SENTINEL ML TRAINING[/bold green]\n"
        f"[dim]Model: {model}[/dim]",
        border_style="green",
    ))

    from sentinel.training.train import (
        train_all, train_threat_mlp, train_threat_lstm,
        train_anomaly_autoencoder, train_reid_net, train_weapon_context,
    )

    fn_map = {
        "all": lambda: train_all(),
        "threat_mlp": lambda: train_threat_mlp(epochs, batch_size, lr),
        "lstm": lambda: train_threat_lstm(epochs, batch_size, lr),
        "anomaly": lambda: train_anomaly_autoencoder(epochs, batch_size, lr),
        "reid": lambda: train_reid_net(epochs, batch_size, lr),
        "weapon": lambda: train_weapon_context(epochs, batch_size, lr),
    }

    fn_map[model]()
    console.print("\n[bold green]Training complete![/bold green]")


@cli.command("generate-data")
@click.option("--samples", default=2000, type=int, help="Samples per class")
def generate_data(samples):
    """Generate synthetic training data."""
    console.print("[green]Generating synthetic training data...[/green]")

    from sentinel.training.data_generator import PoseSampleGenerator, SequenceDataGenerator
    import json

    gen = PoseSampleGenerator()
    data = gen.generate_dataset(samples_per_class=samples)
    gen.save_dataset(data)

    seq_gen = SequenceDataGenerator(sequence_length=30)
    seq_data = seq_gen.generate_dataset(sequences_per_class=samples // 4)
    path = gen.output_dir / "sequence_training_data.json"
    with open(path, "w") as f:
        json.dump(seq_data, f)

    console.print(f"[green]Generated {len(data['labels'])} pose samples[/green]")
    console.print(f"[green]Generated {len(seq_data['labels'])} sequences[/green]")


@cli.command()
def status():
    """Show system status and model availability."""
    from pathlib import Path
    from sentinel.config import settings

    table = Table(title="SENTINEL System Status", border_style="green")
    table.add_column("Component", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    # Models
    models_dir = Path(settings.MODELS_DIR)
    model_files = {
        "Threat MLP": "threat_mlp.pt",
        "Threat LSTM": "threat_lstm.pt",
        "Anomaly Autoencoder": "anomaly_autoencoder.pt",
        "Re-ID Network": "reid_net.pt",
        "Weapon Context": "weapon_context.pt",
        "YOLOv8": settings.YOLO_MODEL,
    }

    for name, fname in model_files.items():
        path = models_dir / fname
        if path.exists():
            size = path.stat().st_size / 1024
            table.add_row(name, "[green]READY[/green]", f"{size:.0f} KB")
        else:
            table.add_row(name, "[red]NOT TRAINED[/red]", "Run: python run.py train")

    # Database
    db_path = Path(settings.DB_PATH)
    if db_path.exists():
        size = db_path.stat().st_size / 1024
        table.add_row("Database", "[green]READY[/green]", f"{size:.0f} KB")
    else:
        table.add_row("Database", "[yellow]EMPTY[/yellow]", "Will be created on first run")

    # Training data
    data_path = Path(settings.TRAINING_DATA_DIR) / "threat_training_data.json"
    if data_path.exists():
        size = data_path.stat().st_size / 1024
        table.add_row("Training Data", "[green]READY[/green]", f"{size:.0f} KB")
    else:
        table.add_row("Training Data", "[yellow]NOT GENERATED[/yellow]", "Run: python run.py generate-data")

    # Recordings
    rec_dir = Path(settings.RECORDINGS_DIR)
    num_recs = len(list(rec_dir.glob("*.avi"))) if rec_dir.exists() else 0
    table.add_row("Recordings", f"[green]{num_recs} clips[/green]", str(rec_dir))

    console.print(table)

    console.print(f"\n[green]Server:[/green] python run.py serve")
    console.print(f"[green]Train:[/green]  python run.py train")
    console.print(f"[green]Data:[/green]   python run.py generate-data")


if __name__ == "__main__":
    cli()
