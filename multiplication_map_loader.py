import numpy as np
from PIL import Image
import os

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
