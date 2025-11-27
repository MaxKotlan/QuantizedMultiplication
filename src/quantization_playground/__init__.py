"""
Quantization playground package.

Provides lookup-table multiplication utilities, map generation, and
simulation helpers.
"""

from .paths import DATA_DIR, MAPS_DIR, SIMULATION_DIR, ensure_data_dirs
from .export_examples import export_simulation_examples

__all__ = ["DATA_DIR", "MAPS_DIR", "SIMULATION_DIR", "ensure_data_dirs", "export_simulation_examples"]
