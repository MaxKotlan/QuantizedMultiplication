import numpy as np
from multiplication_map_loader import MAP_CONFIG

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
