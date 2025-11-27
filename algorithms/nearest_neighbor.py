from multiplication_map_loader import MAP_CONFIG
import numpy as np

def _multiplyIntSpace(a, b, uint8_map):
    size = uint8_map.shape[0]  # assume square
    # a, b are already in map index space
    x = int(np.clip(a, 0, size - 1))
    y = int(np.clip(b, 0, size - 1))
    return uint8_map[x, y]

def multiplyFloatSpaceNN(fa, fb, uint8_map, map_type='signed_ext'):
    min_f, max_f = MAP_CONFIG[map_type]['float_range']
    size = uint8_map.shape[0]
    map_max = np.max(uint8_map)
    scale_in = (size - 1) / (max_f - min_f)

    # Clamp inputs to representable float range to avoid index overflow
    fa = np.clip(fa, min_f, max_f)
    fb = np.clip(fb, min_f, max_f)

    # Map floats to map index space (0..size-1)
    ia = int(np.clip(round((fa - min_f) * scale_in), 0, size - 1))
    ib = int(np.clip(round((fb - min_f) * scale_in), 0, size - 1))
    
    ir = _multiplyIntSpace(ia, ib, uint8_map)

    if map_type == 'signed_log':
        half_max = max(map_max / 2, 1e-9)
        # Decode logarithmic mapping
        fz_log = (ir / half_max) - 1
        sign = np.sign(fz_log)
        abs_val = np.abs(fz_log)
        fz = sign * ((10 ** abs_val - 1) / 9) * 4  # inverse of encoding (fz_norm * 4)
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
