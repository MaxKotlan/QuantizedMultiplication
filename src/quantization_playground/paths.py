"""
Shared filesystem paths for the project.

Defaults point to the repository layout (data/ for generated assets), but can
be overridden via environment variables when embedding elsewhere.
"""
from pathlib import Path
import os


def _env_path(key: str, default: Path) -> Path:
    override = os.environ.get(key)
    return Path(override) if override else default


# .../src/quantization_playground/paths.py -> repo root sits two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = _env_path("QUANTIZATION_DATA_DIR", PROJECT_ROOT / "data")
MAPS_DIR = _env_path("QUANTIZATION_MAPS_DIR", DATA_DIR / "multiplication_maps")
SIMULATION_DIR = _env_path("QUANTIZATION_SIM_DIR", DATA_DIR / "simulation")


def ensure_data_dirs() -> None:
    """Create data folders up front so callers do not need to."""
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    SIMULATION_DIR.mkdir(parents=True, exist_ok=True)

