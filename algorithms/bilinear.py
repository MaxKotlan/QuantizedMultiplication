import numpy as np
from multiplication_map_loader import MAP_CONFIG

def multiplyFloatSpaceInterpolated(fa, fb, uint8_map, map_type='signed', float_range=None):
    min_f, max_f = float_range if float_range else MAP_CONFIG[map_type]['float_range']
    size = uint8_map.shape[0]
    map_max = np.max(uint8_map)
    scale = (size - 1) / (max_f - min_f)

    # Clamp inputs to representable float range to avoid index overflow
    fa = np.clip(fa, min_f, max_f)
    fb = np.clip(fb, min_f, max_f)

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

    if map_type == 'signed_log':
        half_max = max(map_max / 2, 1e-9)
        fz_log = (val / half_max) - 1
        sign = np.sign(fz_log)
        abs_val = np.abs(fz_log)
        max_range = max(abs(min_f), abs(max_f))
        fz = sign * ((10 ** abs_val - 1) / 9) * max_range
        return fz
    else:
        scale_out = map_max / (max_f - min_f)
        return val / scale_out + min_f
