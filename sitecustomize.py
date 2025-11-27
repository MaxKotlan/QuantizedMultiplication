"""
Ensure the repo's src/ directory is importable when running scripts directly.

This lets you run commands like:
    python -m quantization_playground.simulation
without needing a pip install or manual PYTHONPATH export.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
