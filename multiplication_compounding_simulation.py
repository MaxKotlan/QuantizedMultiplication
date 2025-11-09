import numpy as np
from multiplication_map_loader import load_multiplication_map
from test_float import testFloat
from graph import plotChain
import time

def testLongLivingChain(uint8_map, fa_init, fb_seq, map_type='signed_ext', method='interpolated'):
    fr_min, fr_max = -2, 2
    chain_data = []

    regular = fa_init
    mapped = fa_init

    for i, fb in enumerate(fb_seq):
        # Clamp product to map range (soft clamping)
        reg_result = regular * fb
        if reg_result > fr_max:
            scale = fr_max / (reg_result + 1e-12)
            fb *= scale * 0.9 + 0.1
            reg_result = regular * fb
        elif reg_result < fr_min:
            scale = fr_min / (reg_result - 1e-12)
            fb *= scale * 0.9 + 0.1
            reg_result = regular * fb

        fa_val, fb_val, reg_val, map_val, err = testFloat(mapped, fb, uint8_map, map_type=map_type, method=method)
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



if __name__ == "__main__":
    seed = int(time.time()) % (2**32 - 1)
    np.random.seed(seed)  # reproducible sequence across all simulations
    chain_length = 1024

    # Generate the same initial float and sequence for all simulations
    fa_init = np.random.uniform(0.95, 1.05)
    fb_seq = np.random.uniform(0.9, 1.1, size=chain_length)

    # map_sizes = [4, 8, 16, 32, 64, 128, 256]
    map_sizes = [256]
    methods = ['nearest']#, 'interpolated']
    map_types = ['signed_ext', 'signed_log']#, 'signed_ext_warped']  # added signed_log here

    for method in methods:
        for map_type in map_types:
            for size in map_sizes:
                uint8_map = load_multiplication_map(size, map_type=map_type)
                chain, f_reg, f_map, f_abs, f_perc = testLongLivingChain(
                    uint8_map, fa_init, fb_seq, map_type=map_type, method=method
                )

                print(f"\n=== Map type: {map_type}, Size: {size}, Method: {method} ===")
                print(f"Final regular: {f_reg:.6f}, mapped: {f_map:.6f}, abs_err: {f_abs:.6f}, perc_err: {f_perc:.2f}%")

                plotChain(chain, filename=f"chain_plot_{map_type}_{size}_{method}.png")
