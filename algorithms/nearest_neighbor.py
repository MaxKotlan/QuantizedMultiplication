from multiplication_map_loader import MAP_CONFIG

# Keep multiplyIntSpace inside this module, private to nearest neighbor
def _multiplyIntSpace(a, b, uint8_map):
    size = uint8_map.shape[0]  # assume square
    x = int(a * (size-1) / 255)
    y = int(b * (size-1) / 255)
    return uint8_map[x, y]

def multiplyFloatSpaceNN(fa, fb, uint8_map, map_type='signed'):
    min_f, max_f = MAP_CONFIG[map_type]['float_range']
    scale = 255 / (max_f - min_f)
    ia = round((fa - min_f) * scale)
    ib = round((fb - min_f) * scale)
    ir = _multiplyIntSpace(ia, ib, uint8_map)
    return ir / scale + min_f
