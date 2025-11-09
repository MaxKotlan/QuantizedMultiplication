from multiplication_map_loader import MAP_CONFIG
from log_helpers import float_to_log_index, log_index_to_float

def _multiplyIntSpace(a, b, uint8_map):
    size = uint8_map.shape[0]  # assume square
    x = int(a * (size-1) / 255)
    y = int(b * (size-1) / 255)
    return uint8_map[x, y]

def multiplyFloatSpaceNN(fa, fb, uint8_map, map_type='signed'):
    min_f, max_f = MAP_CONFIG[map_type]['float_range']
    size = uint8_map.shape[0]

    if map_type == 'signed_log':
        # map floats to indices in log-space
        ia = float_to_log_index(fa, min_f, max_f, size)
        ib = float_to_log_index(fb, min_f, max_f, size)
        ir = uint8_map[ia, ib]
        # convert back to float using inverse log
        return log_index_to_float(ir, size, min_f, max_f)

    # default linear mapping
    scale = 255 / (max_f - min_f)
    ia = round((fa - min_f) * scale)
    ib = round((fb - min_f) * scale)
    ir = _multiplyIntSpace(ia, ib, uint8_map)
    return ir / scale + min_f
