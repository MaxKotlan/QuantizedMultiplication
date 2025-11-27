from multiplication_map_loader import load_multiplication_map, MAP_CONFIG
from algorithms import nearest_neighbor, bilinear

def testFloat(fa, fb, uint8_map, map_type='signed', method='interpolated', float_range=None):
    regular = fa * fb
    if method == 'nearest':
        mapped_value = nearest_neighbor.multiplyFloatSpaceNN(fa, fb, uint8_map, map_type, float_range=float_range)
    else:
        mapped_value = bilinear.multiplyFloatSpaceInterpolated(fa, fb, uint8_map, map_type, float_range=float_range)

    fr = float_range if float_range else MAP_CONFIG[map_type]['float_range']
    max_abs = max(abs(fr[0]), abs(fr[1]))
    error_percent = abs(mapped_value - regular) * 100 / max_abs
    return fa, fb, regular, mapped_value, error_percent

def printTestFloatResults(results):
    fa, fb, regular, mapped_value, error_percent = results
    print(f"Regular multiplication: {fa} * {fb} = {regular:.6f}")
    print(f"Mapped multiplication: {mapped_value:.6f}")
    print(f"Error (positive %): {error_percent:.2f}%")

# Sanity Check Test
if __name__ == "__main__":
    uint8_map = load_multiplication_map(16, 'signed_ext')

    # Example sanity checks
    tests = [(1.5, -1.0), (0.5, 0.5), (-1.5, 1.0)]
    for fa, fb in tests:
        results = testFloat(fa, fb, uint8_map, 'signed_ext', method='interpolated')
        printTestFloatResults(results)
