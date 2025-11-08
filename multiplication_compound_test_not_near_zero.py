import numpy as np
from multiplication_map_loader import load_multiplication_map, testFloat
import matplotlib.pyplot as plt

uint8_map = load_multiplication_map(256)

def testLongLivingChain(uint8_map, chain_length=50, seed=None):
    """
    Generate a multiplication chain that avoids collapsing to zero.
    Each step picks a multiplier in [0.9, 1.1] to maintain magnitude.
    """
    if seed is not None:
        np.random.seed(seed)

    chain_data = []

    # Start value
    fa = np.random.uniform(0.9, 1.1)
    regular = fa
    mapped = fa

    for i in range(chain_length):
        # Pick next multiplier that keeps the chain alive
        fb = np.random.uniform(0.9, 1.1)

        reg_result = regular * fb
        fa_val, fb_val, reg_val, map_val, err = testFloat(mapped, fb, uint8_map)

        abs_err = abs(map_val - reg_result)
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

    final_reg = regular
    final_map = mapped
    final_abs_error = abs(final_map - final_reg)
    final_perc_error = (final_abs_error / abs(final_reg) * 100) if abs(final_reg) > 1e-6 else 0.0

    return chain_data, final_reg, final_map, final_abs_error, final_perc_error


def plotChain(chain_data, filename="chain_plot.png"):
    steps = [step['step'] for step in chain_data]
    regular = [step['regular_result'] for step in chain_data]
    mapped = [step['mapped_result'] for step in chain_data]
    errors = [step['percent_error'] / 100.0 for step in chain_data]  # scale 0-1

    plt.figure(figsize=(10, 5))
    plt.plot(steps, regular, 'o-', label='Regular float')
    plt.plot(steps, mapped, 's-', label='Mapped multiplication')
    plt.plot(steps, errors, 'r--', label='Percent error (scaled 0-1)')
    
    plt.title("Chained Multiplication: Regular vs Mapped")
    plt.xlabel("Step")
    plt.ylabel("Value / Scaled Percent Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

if __name__ == "__main__":
    chain, f_reg, f_map, f_abs, f_perc = testLongLivingChain(uint8_map, chain_length=20)
    from pprint import pprint
    pprint(chain)
    print(f"Final regular: {f_reg:.6f}, mapped: {f_map:.6f}, abs_err: {f_abs:.6f}, perc_err: {f_perc:.2f}%")
    
    # Plot the chain
    plotChain(chain)
