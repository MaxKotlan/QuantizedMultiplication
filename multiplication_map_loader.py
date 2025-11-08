import numpy as np
from PIL import Image
import os

MAP_CONFIG = {
    'unsigned':   {'folder': './multiplication_maps', 'float_range': (0, 1), 'prefix': 'unsigned'},
    'signed':     {'folder': './multiplication_maps', 'float_range': (-1, 1), 'prefix': 'signed'},
    'signed_ext': {'folder': './multiplication_maps', 'float_range': (-2, 2), 'prefix': 'signed_extended'}
}

def load_multiplication_map(size, map_type='signed'):
    cfg = MAP_CONFIG[map_type]
    path = os.path.join(cfg['folder'], f"{cfg['prefix']}_{size}x{size}.png")
    img = Image.open(path)
    return np.array(img, dtype=np.uint8)

def multiplyIntSpace(a, b, uint8_map):
    size = uint8_map.shape[0]  # assume square
    x = int(a * (size-1) / 255)
    y = int(b * (size-1) / 255)
    return uint8_map[x, y]

# Nearest-neighbor
def multiplyFloatSpaceNN(fa, fb, uint8_map, map_type='signed'):
    min_f, max_f = MAP_CONFIG[map_type]['float_range']
    scale = 255 / (max_f - min_f)
    ia = round((fa - min_f) * scale)
    ib = round((fb - min_f) * scale)
    ir = multiplyIntSpace(ia, ib, uint8_map)
    return ir / scale + min_f

# Bilinear interpolation
def multiplyFloatSpaceInterpolated(fa, fb, uint8_map, map_type='signed'):
    min_f, max_f = MAP_CONFIG[map_type]['float_range']
    size = uint8_map.shape[0]
    scale = (size - 1) / (max_f - min_f)

    x = (fa - min_f) * scale
    y = (fb - min_f) * scale

    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, size - 1), min(y0 + 1, size - 1)

    fx, fy = x - x0, y - y0

    val = (
        uint8_map[x0, y0] * (1 - fx) * (1 - fy) +
        uint8_map[x1, y0] * fx * (1 - fy) +
        uint8_map[x0, y1] * (1 - fx) * fy +
        uint8_map[x1, y1] * fx * fy
    )

    return val * (max_f - min_f) / 255 + min_f

# testFloat supporting both methods
def testFloat(fa, fb, uint8_map, map_type='signed', method='interpolated'):
    regular = fa * fb
    if method == 'nearest':
        mapped_value = multiplyFloatSpaceNN(fa, fb, uint8_map, map_type)
    else:
        mapped_value = multiplyFloatSpaceInterpolated(fa, fb, uint8_map, map_type)

    fr = MAP_CONFIG[map_type]['float_range']
    max_abs = max(abs(fr[0]), abs(fr[1]))
    error_percent = abs(mapped_value - regular) * 100 / max_abs
    return fa, fb, regular, mapped_value, error_percent

def printTestFloatResults(results):
    fa, fb, regular, mapped_value, error_percent = results
    print(f"Regular multiplication: {fa} * {fb} = {regular:.6f}")
    print(f"Mapped multiplication: {mapped_value:.6f}")
    print(f"Error (positive %): {error_percent:.2f}%")

# Example usage
if __name__ == "__main__":
    uint8_map = load_multiplication_map(16, 'signed_ext')
    results = testFloat(1.5, -1.0, uint8_map, 'signed_ext', method='interpolated')
    printTestFloatResults(results)
