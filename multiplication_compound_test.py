import numpy as np
from multiplication_map_loader import load_multiplication_map, testFloat

uint8_map = load_multiplication_map(256)

def testCompoundingMultiplications(uint8_map, chain_length=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    fa = np.random.uniform(-1.0, 1.0)
    chain_data = []

    regular = fa
    mapped = fa

    for i in range(chain_length):
        fb = np.random.uniform(-1.0, 1.0)
        reg_result = regular * fb
        fa_val, fb_val, reg_val, map_val, err = testFloat(mapped, fb, uint8_map)
        chain_data.append({
            "step": i,
            "fa": fa_val,
            "fb": fb_val,
            "regular_result": reg_result,
            "mapped_result": map_val,
            "error_percent": err
        })
        regular = reg_result
        mapped = map_val

    # final results
    final_reg = regular
    final_map = mapped
    final_error = abs((final_map - final_reg) / final_reg) * 100 if final_reg != 0 else 0
    return chain_data, final_reg, final_map, final_error

def printChain(chain_data, final_reg, final_map, final_error):
    for step in chain_data:
        print(
            f"Step {step['step']:2d} | "
            f"fa={step['fa']:+.6f}, fb={step['fb']:+.6f} | "
            f"reg={step['regular_result']:+.6f}, "
            f"map={step['mapped_result']:+.6f} | "
            f"err={step['error_percent']:.2f}%"
        )
    print(f"\nFinal regular result: {final_reg:+.6f}")
    print(f"Final mapped result : {final_map:+.6f}")
    print(f"Final compounded error: {final_error:.2f}%")

if __name__ == "__main__":
    chain, final_reg, final_map, final_error = testCompoundingMultiplications(uint8_map, chain_length=3)
    printChain(chain, final_reg, final_map, final_error)
