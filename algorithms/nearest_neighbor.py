from multiplication_map_loader import MAP_CONFIG
import numpy as np

def _multiplyIntSpace(a, b, uint8_map):
    size = uint8_map.shape[0]  # assume square
    # a, b are already in map index space
    x = int(np.clip(a, 0, size - 1))
    y = int(np.clip(b, 0, size - 1))
    return uint8_map[x, y]

def multiplyFloatSpaceNN(fa, fb, uint8_map, map_type='signed_ext', float_range=None, stochastic_round=False):
    min_f, max_f = float_range if float_range else MAP_CONFIG[map_type]['float_range']
    size = uint8_map.shape[0]
    map_max = np.max(uint8_map)
    scale_in = (size - 1) / (max_f - min_f)

    # Clamp inputs to representable float range to avoid index overflow
    fa = np.clip(fa, min_f, max_f)
    fb = np.clip(fb, min_f, max_f)

    # Map floats to map index space (0..size-1)
    if stochastic_round:
        ia = int(np.clip(np.floor((fa - min_f) * scale_in + np.random.random()), 0, size - 1))
        ib = int(np.clip(np.floor((fb - min_f) * scale_in + np.random.random()), 0, size - 1))
    else:
        ia = int(np.clip(round((fa - min_f) * scale_in), 0, size - 1))
        ib = int(np.clip(round((fb - min_f) * scale_in), 0, size - 1))
    
    ir = _multiplyIntSpace(ia, ib, uint8_map)

    if map_type == 'signed_log':
        max_range = max(abs(min_f), abs(max_f))
        half_max = max(map_max / 2, 1e-9)
        # Decode logarithmic mapping
        fz_log = (ir / half_max) - 1
        sign = np.sign(fz_log)
        abs_val = np.abs(fz_log)
        fz = sign * ((10 ** abs_val - 1) / 9) * max_range  # inverse of encoding (fz_norm * max_range)
        return fz
    elif map_type == 'signed_ext_warp':
        k = 20.0
        fr_min, fr_max = MAP_CONFIG[map_type]['float_range']
        fz_norm = (ir / 127.5) - 1
        sign = np.sign(fz_norm)
        abs_val = np.abs(fz_norm)
        fz = sign * np.sinh(abs_val * np.arcsinh(k * (fr_max ** 2))) / k
        return fz  # <-- don't rescale here
    else:
        # Linear decode
        scale_out = map_max / (max_f - min_f)
        return ir / scale_out + min_f
