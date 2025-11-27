from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image

from ..paths import MAPS_DIR

MAP_CONFIG = {
    "unsigned": {"folder": MAPS_DIR, "float_range": (0, 1), "prefix": "unsigned"},
    "signed": {"folder": MAPS_DIR, "float_range": (-1, 1), "prefix": "signed"},
    "signed_ext": {"folder": MAPS_DIR, "float_range": (-2, 2), "prefix": "signed_extended"},
    "signed_log": {"folder": MAPS_DIR, "float_range": (-2, 2), "prefix": "signed_log"},
    "signed_ext_warped": {"folder": MAPS_DIR, "float_range": (-2, 2), "prefix": "signed_extended_warped"},
}


def _bit_depth(size: int) -> int:
    return int(np.ceil(np.log2(size))) if size > 0 else 0


def _map_path(size: int, cfg: dict, suffix: str = "") -> Path:
    folder = Path(cfg["folder"])
    bit_dir = folder / f"{_bit_depth(size)}bit"
    bit_dir.mkdir(parents=True, exist_ok=True)
    return bit_dir / f"{cfg['prefix']}_{size}x{size}{suffix}.png"


def load_multiplication_map(size: int, map_type: str = "signed", suffix: str = "") -> np.ndarray:
    """Load a single multiplication map into memory."""
    cfg = MAP_CONFIG[map_type]
    path = _map_path(size, cfg, suffix)
    img = Image.open(path)
    return np.array(img)


def ensure_multiplication_maps(map_sizes: Iterable[int], map_type: str = "signed", max_range: float = 2.0) -> dict[int, np.ndarray]:
    """
    Ensure the required multiplication maps exist for the given sizes/type/range.
    If missing for the current max_range, regenerate all maps for that range.
    """
    from .generator import main as generate_maps

    cfg = MAP_CONFIG[map_type]
    suffix = f"_r{max_range}".replace(".", "_")
    expected_files = [_map_path(size, cfg, suffix) for size in map_sizes]
    needs_regen = any(not path.exists() for path in expected_files)

    if needs_regen:
        generate_maps(max_range=max_range, suffix=suffix, output_dir=cfg["folder"])

    return {
        size: np.array(Image.open(path))
        for size, path in zip(map_sizes, expected_files)
    }
