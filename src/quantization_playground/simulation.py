import argparse
import time
import numpy as np

from .maps import MAP_CONFIG, ensure_multiplication_maps
from .evaluation import evaluate_float
from .plotting import plotChain


def _detect_float8_dtype():
    """Return a usable float8 dtype if NumPy supports it, else None."""
    candidates = (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    for name in candidates:
        try:
            return np.dtype(name)
        except TypeError:
            continue
    return None


FLOAT8_DTYPE = _detect_float8_dtype()
BASELINE_CHOICES = ["float8", "float16", "float32", "float64"]


def _resolve_baseline_dtype(name: str) -> tuple[np.dtype, str]:
    """
    Resolve the desired baseline dtype.

    Returns (dtype, label). If float8 is requested but unsupported, falls back
    to float16 and labels the run accordingly.
    """
    if name == "float8":
        if FLOAT8_DTYPE:
            return FLOAT8_DTYPE, "float8"
        print("float8 not supported by this NumPy build; falling back to float16.")
        return np.dtype("float16"), "float8 (via float16 fallback)"
    dtype = np.dtype(name)
    return dtype, dtype.name


def _value_bits_for_size(size: int) -> int:
    """Effective value bits given the discrete levels in the table."""
    return int(np.ceil(np.log2(size))) if size > 0 else 0


def testLongLivingChain(uint8_map, fa_init, fb_seq, map_type='signed_ext', method='interpolated', float_range=None, stochastic_round=False, baseline_dtype=np.float32):
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

        fa_val, fb_val, reg_val, map_val, err = evaluate_float(
            mapped,
            fb,
            uint8_map,
            map_type=map_type,
            method=method,
            float_range=float_range,
            stochastic_round=stochastic_round,
            baseline_dtype=baseline_dtype,
        )
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


def _saturate(val, fr_min, fr_max):
    """Clamp a value to the representable float range."""
    return np.clip(val, fr_min, fr_max)


def testSumOfProducts(uint8_map, fa_seq, fb_seq, map_type='signed_ext', method='interpolated', float_range=None, stochastic_round=False, baseline_dtype=np.float32):
    """
    Simulate multiply-then-accumulate (dot product) using the lookup table.

    Products and the running sum are saturated to the map's float range to mimic
    bounded quantized math (similar to matrix-multiply accumulators with a fixed range).
    """
    fr_min, fr_max = float_range if float_range else MAP_CONFIG[map_type]['float_range']
    float_range = (fr_min, fr_max)
    chain_data = []

    regular_sum = baseline_dtype.type(0.0)
    mapped_sum = baseline_dtype.type(0.0)
    prod_clamps = 0
    sum_clamps = 0

    for i, (fa, fb) in enumerate(zip(fa_seq, fb_seq)):
        fa_val, fb_val, reg_prod, map_prod, _ = evaluate_float(
            fa,
            fb,
            uint8_map,
            map_type=map_type,
            method=method,
            float_range=float_range,
            stochastic_round=stochastic_round,
            baseline_dtype=baseline_dtype,
        )
        reg_prod_sat = _saturate(reg_prod, fr_min, fr_max)
        map_prod_sat = _saturate(map_prod, fr_min, fr_max)
        if reg_prod_sat != reg_prod or map_prod_sat != map_prod:
            prod_clamps += 1

        regular_sum = baseline_dtype.type(regular_sum + reg_prod_sat)
        mapped_sum = baseline_dtype.type(mapped_sum + map_prod_sat)

        reg_sum_sat = _saturate(regular_sum, fr_min, fr_max)
        map_sum_sat = _saturate(mapped_sum, fr_min, fr_max)
        if reg_sum_sat != regular_sum or map_sum_sat != mapped_sum:
            sum_clamps += 1

        regular_sum = baseline_dtype.type(reg_sum_sat)
        mapped_sum = baseline_dtype.type(map_sum_sat)

        abs_err = abs(mapped_sum - regular_sum)
        perc_err = (abs_err / abs(regular_sum) * 100) if abs(regular_sum) > 1e-6 else 0.0

        chain_data.append({
            "step": i,
            "fa": fa_val,
            "fb": fb_val,
            "regular_product": reg_prod,
            "mapped_product": map_prod,
            "regular_result": regular_sum,
            "mapped_result": mapped_sum,
            "abs_error": abs_err,
            "percent_error": perc_err
        })

    final_abs_error = abs(mapped_sum - regular_sum)
    final_perc_error = (final_abs_error / abs(regular_sum) * 100) if abs(regular_sum) > 1e-6 else 0.0

    stats = {"product_saturations": prod_clamps, "sum_saturations": sum_clamps}
    return chain_data, regular_sum, mapped_sum, final_abs_error, final_perc_error, stats


def run_simulation(max_range: float, steps: int, baseline_dtype_name: str = "float16", show_error: bool = False, simulation_type: str = "chain") -> None:
    float_range = (-max_range, max_range)
    baseline_dtype, baseline_label = _resolve_baseline_dtype(baseline_dtype_name)
    simulation_type = simulation_type.lower()
    if simulation_type == "matmul":
        simulation_type = "dot"
    if simulation_type not in {"chain", "dot"}:
        raise ValueError(f"Unsupported simulation_type '{simulation_type}'. Use 'chain', 'dot', or 'matmul'.")

    seed = time.time_ns() % (2**32 - 1)  # new seed each script run, shared by all variants
    np.random.seed(seed)
    print(f"Using seed: {seed}, max_range: {max_range}, simulation_type: {simulation_type}")
    chain_length = int(steps)

    # Generate the same initial float and sequence for all simulations,
    # scaled to stay inside the chosen float_range (helps when max_range < 1)
    base_mag = 0.5 * max_range  # center of walk
    init_jitter = 0.05
    fa_init = baseline_dtype.type(np.random.uniform(base_mag * (1 - init_jitter), base_mag * (1 + init_jitter)))

    jitter = min(0.1, 0.2 * max_range)  # shrink step size when range is small
    fb_seq = np.random.uniform(1 - jitter, 1 + jitter, size=chain_length).astype(baseline_dtype)
    dot_seq_a = np.random.uniform(-base_mag, base_mag, size=chain_length).astype(baseline_dtype)
    dot_seq_b = np.random.uniform(-base_mag, base_mag, size=chain_length).astype(baseline_dtype)

    map_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    methods = ['nearest', 'interpolated']
    map_types = ['signed_ext', 'signed_log']

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
                if simulation_type == "chain":
                    chain, f_reg, f_map, f_abs, f_perc = testLongLivingChain(
                        uint8_map, fa_init, fb_seq, map_type=map_type, method=method, float_range=float_range, baseline_dtype=baseline_dtype
                    )
                    ylabel = "Accumulated product"
                    filename_prefix = "chain"
                    mode_label = "Chain multiply"
                else:
                    chain, f_reg, f_map, f_abs, f_perc, stats = testSumOfProducts(
                        uint8_map, dot_seq_a, dot_seq_b, map_type=map_type, method=method, float_range=float_range, baseline_dtype=baseline_dtype
                    )
                    ylabel = "Accumulated sum"
                    filename_prefix = "dot"
                    mode_label = "Sum of products (matmul-like)"

                print(f"\n=== [{mode_label}] Map type: {map_type}, Size: {size}, Method: {method} ===")
                print(f"Final regular: {f_reg:.6f}, mapped: {f_map:.6f}, abs_err: {f_abs:.6f}, perc_err: {f_perc:.2f}%")
                if simulation_type == "dot":
                    print(f"Saturations — products: {stats['product_saturations']}, sums: {stats['sum_saturations']}")

                value_bits = _value_bits_for_size(size)
                legend_labels = {
                    "float": f"Baseline {baseline_label}",
                    "mapped": f"Int lookup {map_type} {size}x{size} ({method}) ~{value_bits}-bit grid",
                }
                plot_title = f"{mode_label}: {map_type} {size}x{size} int lookup (~{value_bits}-bit, {method}) vs {baseline_label}"
                lookup_label = f"Int lookup {map_type} {size}x{size} ({method}) ~{value_bits}-bit grid"
                filename = f"{value_bits}bit/{filename_prefix}_plot_{map_type}_{size}_{method}.png"
                plotChain(
                    chain,
                    filename=filename,
                    float_range=float_range,
                    title=plot_title,
                    legend_labels=legend_labels,
                    baseline_label=f"Baseline {baseline_label}",
                    lookup_label=lookup_label,
                    value_bits=value_bits,
                    show_error=show_error,
                    ylabel=ylabel,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiplication simulations (chained multiply or multiply-accumulate).")
    parser.add_argument("--max-range", type=float, default=2.0, help="Max magnitude of representable float range (symmetric ±max_range).")
    parser.add_argument("--steps", type=int, default=1024, help="Number of chain steps to run.")
    parser.add_argument("--baseline-dtype", type=str, default="float16", choices=BASELINE_CHOICES, help="Precision used for the reference multiply (float8 falls back to float16 if unsupported).")
    parser.add_argument("--show-error", action="store_true", help="Include percent error subplot in saved figures.")
    parser.add_argument("--simulation-type", type=str, default="chain", choices=["chain", "dot", "matmul"], help="Choose chained multiplies (chain) or multiply-accumulate (dot/matmul) simulation.")
    args = parser.parse_args()

    run_simulation(
        max_range=float(args.max_range),
        steps=int(args.steps),
        baseline_dtype_name=args.baseline_dtype,
        show_error=args.show_error,
        simulation_type=args.simulation_type,
    )


if __name__ == "__main__":
    main()
