from multiplication_map_loader import MAP_CONFIG
import numpy as np

def _multiplyIntSpace(a, b, uint8_map):
    size = uint8_map.shape[0]  # assume square
    x = int(np.clip(a * (size - 1) / 255, 0, size - 1))
    y = int(np.clip(b * (size - 1) / 255, 0, size - 1))
    return uint8_map[x, y]

def multiplyFloatSpaceNN(fa, fb, uint8_map, map_type='signed_ext'):
    min_f, max_f = MAP_CONFIG[map_type]['float_range']
    scale = 255 / (max_f - min_f)

    # Map floats to uint8 input indices
    ia = int(np.clip(round((fa - min_f) * scale), 0, 255))
    ib = int(np.clip(round((fb - min_f) * scale), 0, 255))
    
    ir = _multiplyIntSpace(ia, ib, uint8_map)

    if map_type == 'signed_log':
        # Decode logarithmic mapping
        fz_log = (ir / 127.5) - 1
        sign = np.sign(fz_log)
        abs_val = np.abs(fz_log)
        fz = sign * ((10 ** abs_val - 1) / 9) * 4  # inverse of encoding
        return fz
    else:
        # Linear decode
        return ir / scale + min_f
