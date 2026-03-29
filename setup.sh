#!/bin/bash
# SENTINEL Setup Script
# Creates virtual environment and installs all dependencies

set -e

echo "========================================"
echo "  SENTINEL — Setup Script"
echo "========================================"

# Find a PyTorch-compatible Python (3.10-3.12)
PYTHON=""
for v in python3.12 python3.11 python3.10; do
    if command -v $v &>/dev/null; then
        PYTHON=$v
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10-3.12 required (PyTorch doesn't support 3.13+)"
    echo "Install with: brew install python@3.12"
    exit 1
fi

echo "Using: $PYTHON ($($PYTHON --version))"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "[1/4] Virtual environment exists"
fi

# Activate
source venv/bin/activate

# Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[3/4] Installing dependencies..."
pip install -r requirements.txt

# Initialize database
echo "[4/4] Initializing database..."
python -c "from sentinel.models.database import init_db; init_db()"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  Quick Start:"
echo "    source venv/bin/activate"
echo ""
echo "    # 1. Generate training data"
echo "    python run.py generate-data"
echo ""
echo "    # 2. Train ML models"
echo "    python run.py train"
echo ""
echo "    # 3. Start server"
echo "    python run.py serve"
echo ""
echo "    # Open http://localhost:8099"
echo ""
echo "  Or run standalone (no backend):"
echo "    open sentinel.html"
echo "========================================"
