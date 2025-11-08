import numpy as np
from multiplication_map_loader import load_multiplication_map, testFloat, printTestFloatResults   # import the map from maps.py

uint8_map = load_multiplication_map(16)

def testRandomMultiplications(uint8_map, num_tests=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    results_list = []
    for i in range(num_tests):
        fa = np.random.uniform(-1.0, 1.0)
        fb = np.random.uniform(-1.0, 1.0)
        results = testFloat(fa, fb, uint8_map)
        results_list.append(results)
    return results_list

def printRandomMultiplicationResults(results_list):
    for i, results in enumerate(results_list, 1):
        print(f"\nTest {i}:")
        printTestFloatResults(results)

# Example usage:
results_list = testRandomMultiplications(uint8_map, num_tests=5, seed=42)
printRandomMultiplicationResults(results_list)
