import numpy as np
from PIL import Image
import os
from typing import Tuple

MAP_CONFIG = {
    'unsigned':   {'folder': './multiplication_maps', 'float_range': (0, 1), 'prefix': 'unsigned'},
    'signed':     {'folder': './multiplication_maps', 'float_range': (-1, 1), 'prefix': 'signed'},
    'signed_ext': {'folder': './multiplication_maps', 'float_range': (-2, 2), 'prefix': 'signed_extended'},
    'signed_log': {'folder': './multiplication_maps', 'float_range': (-2, 2), 'prefix': 'signed_log'},
    'signed_ext_warped': {'folder': './multiplication_maps', 'float_range': (-2, 2), 'prefix': 'signed_extended_warped'}
}

def load_multiplication_map(size, map_type='signed'):
    cfg = MAP_CONFIG[map_type]
    path = os.path.join(cfg['folder'], f"{cfg['prefix']}_{size}x{size}.png")
    img = Image.open(path)
    return np.array(img, dtype=np.uint8)


def ensure_multiplication_maps(map_sizes, map_type='signed', max_range=2.0):
    """
    Ensure the required multiplication maps exist for the given sizes/type/range.
    If missing for the current max_range, regenerate all maps for that range.
    """
    from multiplication_map_generator import main as generate_maps

    # Check one representative file for this range; if missing, regen all
    cfg = MAP_CONFIG[map_type]
    folder = cfg['folder']
    os.makedirs(folder, exist_ok=True)

    # We bake max_range into filenames via suffix to avoid stale reuse
    suffix = f"_r{max_range}".replace('.', '_')
    expected_files = [os.path.join(folder, f"{cfg['prefix']}_{size}x{size}{suffix}.png") for size in map_sizes]
    needs_regen = any(not os.path.exists(p) for p in expected_files)

    if needs_regen:
        generate_maps(max_range=max_range, suffix=suffix)

    # Load maps into a dict keyed by size
    maps = {}
    for size, path in zip(map_sizes, expected_files):
        img = Image.open(path)
        maps[size] = np.array(img, dtype=np.uint8)
    return maps
