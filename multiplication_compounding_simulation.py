import argparse
import numpy as np
from multiplication_map_loader import MAP_CONFIG, ensure_multiplication_maps
from test_float import testFloat
from graph import plotChain
import time

def testLongLivingChain(uint8_map, fa_init, fb_seq, map_type='signed_ext', method='interpolated', float_range=None):
    fr_min, fr_max = float_range if float_range else MAP_CONFIG[map_type]['float_range']
    float_range = (fr_min, fr_max)
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

        fa_val, fb_val, reg_val, map_val, err = testFloat(mapped, fb, uint8_map, map_type=map_type, method=method, float_range=float_range)
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
    parser = argparse.ArgumentParser(description="Run chained multiplication simulations.")
    parser.add_argument("--max-range", type=float, default=2.0, help="Max magnitude of representable float range (symmetric ±max_range).")
    args = parser.parse_args()

    max_range = float(args.max_range)
    float_range = (-max_range, max_range)

    seed = time.time_ns() % (2**32 - 1)  # new seed each script run, shared by all variants
    np.random.seed(seed)
    print(f"Using seed: {seed}, max_range: {max_range}")
    chain_length = 1024

    # Generate the same initial float and sequence for all simulations
    fa_init = np.random.uniform(0.95, 1.05)
    fb_seq = np.random.uniform(0.9, 1.1, size=chain_length)

    map_sizes = [4, 8, 16, 32, 64, 128, 256]
    methods = ['nearest', 'interpolated']
    map_types = ['signed_ext', 'signed_log']#, 'signed_ext_warped']  # added signed_log here

    # Ensure maps exist (with suffix tied to max_range) and load them
    suffix = f"_r{max_range}".replace('.', '_')
    maps_by_type = {
        mt: ensure_multiplication_maps(map_sizes, map_type=mt, max_range=max_range)
        for mt in map_types
    }

    for method in methods:
        for map_type in map_types:
            for size in map_sizes:
                uint8_map = maps_by_type[map_type][size]
                chain, f_reg, f_map, f_abs, f_perc = testLongLivingChain(
                    uint8_map, fa_init, fb_seq, map_type=map_type, method=method, float_range=float_range
                )

                print(f"\n=== Map type: {map_type}, Size: {size}, Method: {method} ===")
                print(f"Final regular: {f_reg:.6f}, mapped: {f_map:.6f}, abs_err: {f_abs:.6f}, perc_err: {f_perc:.2f}%")

                plotChain(chain, filename=f"chain_plot_{map_type}_{size}_{method}.png", float_range=float_range)
