import numpy as np
from multiplication_map_loader import load_multiplication_map, testFloat

uint8_map = load_multiplication_map(256)

def testRandomMultiplications(uint8_map, num_tests=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    error_rates = []
    for _ in range(num_tests):
        fa = np.random.uniform(-1.0, 1.0)
        fb = np.random.uniform(-1.0, 1.0)
        results = testFloat(fa, fb, uint8_map)
        # results = (fa, fb, regular, mapped_value, error_percent)
        error_rates.append(results[4])  # directly use the positive error rate
    return np.array(error_rates)

def printErrorStatistics(error_rates):
    print(f"Number of tests: {len(error_rates)}")
    print(f"Average error (positive %): {np.mean(error_rates):.2f}%")
    print(f"Minimum error (positive %): {np.min(error_rates):.2f}%")
    print(f"Maximum error (positive %): {np.max(error_rates):.2f}%")

if __name__ == "__main__":
    errors = testRandomMultiplications(uint8_map, num_tests=10000, seed=42)
    printErrorStatistics(errors)
