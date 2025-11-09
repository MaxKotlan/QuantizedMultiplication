import numpy as np
from multiplication_map_loader import load_multiplication_map, testFloat

uint8_map = load_multiplication_map(16)

def testCompoundingMultiplications(uint8_map, chain_length=10, values=None, random_range=(0.5, 1.0), seed=None):
    """
    Performs a chained multiplication test comparing mapped vs regular float multiplication.

    - values: optional list/array of floats to multiply (fixed sequence)
    - random_range: tuple (min, max) for generating random floats if values not provided
    """
    if seed is not None:
        np.random.seed(seed)

    chain_data = []

    # Initialize first value
    if values is not None and len(values) > 0:
        fa = values[0]
    else:
        fa = np.random.uniform(*random_range)

    regular = fa
    mapped = fa

    for i in range(chain_length):
        # Get next multiplier
        if values is not None and i+1 < len(values):
            fb = values[i+1]
        else:
            fb = np.random.uniform(*random_range)

        # Regular float multiplication
        reg_result = regular * fb

        # Mapped multiplication
        fa_val, fb_val, reg_val, map_val, err = testFloat(mapped, fb, uint8_map)

        # Absolute error
        abs_err = abs(map_val - reg_result)
        # Percent error, avoid dividing by near-zero
        perc_err = (abs_err / abs(reg_result) * 100) if abs(reg_result) > 1e-6 else 0.0

        chain_data.append({
            "step": i,
            "fa": fa_val,
            "fb": fb_val,
            "regular_result": reg_result,
            "mapped_result": map_val,
            "abs_error": abs_err,
            "percent_error": perc_err
        })

        regular = reg_result
        mapped = map_val

    # Final results
    final_reg = regular
    final_map = mapped
    final_abs_error = abs(final_map - final_reg)
    final_perc_error = (final_abs_error / abs(final_reg) * 100) if abs(final_reg) > 1e-6 else 0.0

    return chain_data, final_reg, final_map, final_abs_error, final_perc_error

def printChain(chain_data, final_reg, final_map, final_abs_error, final_perc_error):
    for step in chain_data:
        print(
            f"Step {step['step']:2d} | "
            f"fa={step['fa']:+.6f}, fb={step['fb']:+.6f} | "
            f"reg={step['regular_result']:+.6f}, "
            f"map={step['mapped_result']:+.6f} | "
            f"abs_err={step['abs_error']:.6f}, "
            f"perc_err={step['percent_error']:.2f}%"
        )
    print(f"\nFinal regular result: {final_reg:+.6f}")
    print(f"Final mapped result : {final_map:+.6f}")
    print(f"Final absolute error: {final_abs_error:.6f}")
    print(f"Final percent error : {final_perc_error:.2f}%")

if __name__ == "__main__":
    # Example 1: fixed chain of 0.9 * 0.9 * 0.9
    values = [0.9, 0.9, 0.9]
    chain, final_reg, final_map, final_abs_error, final_perc_error = \
        testCompoundingMultiplications(uint8_map, chain_length=len(values), values=values)
    printChain(chain, final_reg, final_map, final_abs_error, final_perc_error)

    # Example 2: random chain avoiding near-zero drift
    chain2, f_reg2, f_map2, f_abs2, f_perc2 = \
        testCompoundingMultiplications(uint8_map, chain_length=67, random_range=(0.5,1.0), seed=42)
    print("\nRandom chain test:")
    printChain(chain2, f_reg2, f_map2, f_abs2, f_perc2)
