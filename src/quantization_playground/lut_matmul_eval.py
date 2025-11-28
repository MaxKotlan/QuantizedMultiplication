"""
Evaluate LUT-based multiply-accumulate on a real model tensor (GGUF) to gauge error.

Example:
    MPLCONFIGDIR="$PWD/.mplconfig" . .venv/bin/activate && \
    python -m quantization_playground.lut_matmul_eval \
        --model /var/lib/ollama/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29 \
        --tensor blk.0.attn_q_proj.weight \
        --rows 4 --cols 512 \
        --simulation-type matmul \
        --baseline-dtype float16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from gguf import GGUFReader

from .maps import ensure_multiplication_maps, MAP_CONFIG
from .simulation import BASELINE_CHOICES, _resolve_baseline_dtype, testSumOfProducts, _value_bits_for_size
from .paths import SIMULATION_DIR
from .plotting import plotChain


def _load_tensor(reader: GGUFReader, name: str) -> np.ndarray:
    for t in reader.tensors:
        if t.name == name:
            return np.array(t.data)  # loads into memory
    raise KeyError(f"Tensor '{name}' not found. Use --list-tensors to inspect available names.")


def _list_tensors(reader: GGUFReader, limit: int | None = None) -> None:
    total = len(reader.tensors)
    for i, t in enumerate(reader.tensors):
        if limit is not None and i >= limit:
            print(f"... ({total - limit} more)")
            break
        print(f"{t.name}: shape={t.data.shape}, dtype={t.data.dtype}")


def _select_tensor_slice(tensor: np.ndarray, rows: int | None, cols: int | None) -> np.ndarray:
    r = rows if rows is not None else tensor.shape[0]
    c = cols if cols is not None else tensor.shape[1]
    return tensor[:r, :c]


def _model_name(reader: GGUFReader, model_path: Path) -> str:
    try:
        field = reader.get_field("general.name")
        return bytes(field.parts[-1]).decode("utf-8")
    except Exception:
        return model_path.name


def _parse_map_sizes(arg: str) -> list[int]:
    if arg.strip().lower() == "auto":
        return [4, 8, 16, 32, 64, 128, 256, 512]
    sizes = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    return sizes


def _auto_range(weight_slice: np.ndarray, input_vec: np.ndarray, margin: float = 0.1, fallback: float = 1.0) -> float:
    max_w = float(np.max(np.abs(weight_slice))) if weight_slice.size else 0.0
    max_in = float(np.max(np.abs(input_vec))) if input_vec.size else 0.0
    base = max(max_w, max_in, 0.0)
    if base <= 0:
        return fallback
    return base * (1.0 + margin)


def _generate_input(cols: int, baseline_dtype: np.dtype, input_path: Path | None = None) -> np.ndarray:
    if input_path:
        vec = np.load(input_path)
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D input vector, got shape {vec.shape}")
        if vec.shape[0] < cols:
            raise ValueError(f"Input vector length {vec.shape[0]} is smaller than requested cols {cols}")
        return vec[:cols].astype(baseline_dtype, copy=False)
    # Zero-mean small variance to avoid immediate saturation; tweak scale if needed.
    return np.random.normal(loc=0.0, scale=0.3, size=cols).astype(baseline_dtype)


def evaluate_tensor_rows(
    weight_slice: np.ndarray,
    input_vec: np.ndarray,
    uint8_map: np.ndarray,
    map_type: str,
    method: str,
    float_range: tuple[float, float],
    baseline_dtype: np.dtype,
) -> list[dict]:
    rows = []
    for i, row in enumerate(weight_slice):
        chain, f_reg, f_map, f_abs, f_perc, stats = testSumOfProducts(
            uint8_map,
            input_vec,
            row.astype(baseline_dtype),
            map_type=map_type,
            method=method,
            float_range=float_range,
            baseline_dtype=baseline_dtype,
        )
        rows.append(
            {
                "row": i,
                "ref": float(f_reg),
                "mapped": float(f_map),
                "abs_error": float(f_abs),
                "percent_error": float(f_perc),
                "product_saturations": stats["product_saturations"],
                "sum_saturations": stats["sum_saturations"],
                "chain": chain,
            }
        )
    return rows


def summarize(rows: Sequence[dict]) -> dict:
    if not rows:
        return {}
    abs_errors = [r["abs_error"] for r in rows]
    perc_errors = [r["percent_error"] for r in rows]
    prod_sat = sum(r["product_saturations"] for r in rows)
    sum_sat = sum(r["sum_saturations"] for r in rows)
    return {
        "mean_abs_error": float(np.mean(abs_errors)),
        "max_abs_error": float(np.max(abs_errors)),
        "mean_percent_error": float(np.mean(perc_errors)),
        "max_percent_error": float(np.max(perc_errors)),
        "total_product_saturations": int(prod_sat),
        "total_sum_saturations": int(sum_sat),
    }


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LUT-based matmul on a GGUF tensor.")
    parser.add_argument("--model", type=Path, required=True, help="Path to GGUF model blob.")
    parser.add_argument("--tensor", type=str, help="Tensor name to evaluate (see --list-tensors).")
    parser.add_argument("--list-tensors", action="store_true", help="List tensor names and exit.")
    parser.add_argument("--list-limit", type=int, default=30, help="Limit number of tensors to list (default 30).")
    parser.add_argument("--rows", type=int, default=1, help="Number of rows to evaluate from the tensor (default 1).")
    parser.add_argument("--cols", type=int, default=None, help="Number of columns to use (prefix). Defaults to full width.")
    parser.add_argument("--input", type=Path, default=None, help="Optional .npy input vector. If omitted, a random vector is generated.")
    parser.add_argument("--max-range", type=float, default=2.0, help="Float range ±R for the LUT maps (ignored if --auto-max-range).")
    parser.add_argument("--auto-max-range", action="store_true", help="Derive range from tensor/input magnitudes (adds ~10 percent margin).")
    parser.add_argument("--method", type=str, default="both", choices=["nearest", "interpolated", "both"], help="Lookup method (or 'both' to run both).")
    parser.add_argument("--map-type", type=str, default="all", choices=list(MAP_CONFIG.keys()) + ["all"], help="Map encoding to use (or 'all' to run signed_ext and signed_log).")
    parser.add_argument("--map-sizes", type=str, default="auto", help="Comma-separated map sizes (e.g., 16,32,64). Use 'auto' to run 4,8,16,32,64,128,256,512.")
    parser.add_argument("--baseline-dtype", type=str, default="float16", choices=BASELINE_CHOICES, help="Reference dtype for the float matmul.")
    parser.add_argument("--simulation-type", type=str, default="matmul", choices=["matmul", "dot"], help="Alias to match the simulation naming.")
    parser.add_argument("--plot", action="store_true", help="Save simulation-style per-row chain plots (regular vs mapped).")
    parser.add_argument("--plot-show-error", action="store_true", help="Include percent error subplot in the saved plots.")
    args = parser.parse_args()

    model_path = Path(args.model)
    reader = GGUFReader(model_path)
    if args.list_tensors or not args.tensor:
        _list_tensors(reader, limit=args.list_limit)
        if not args.tensor:
            return

    tensor = _load_tensor(reader, args.tensor)
    if tensor.ndim != 2:
        raise ValueError(f"Tensor '{args.tensor}' is not 2D (got shape {tensor.shape})")

    weight_slice = _select_tensor_slice(tensor, args.rows, args.cols)
    _, baseline_label = _resolve_baseline_dtype(args.baseline_dtype)
    baseline_dtype = np.dtype(args.baseline_dtype)

    input_vec = _generate_input(weight_slice.shape[1], baseline_dtype, input_path=args.input)

    max_range = _auto_range(weight_slice, input_vec) if args.auto_max_range else args.max_range
    float_range = (-max_range, max_range)
    map_sizes = _parse_map_sizes(args.map_sizes)
    map_types = ["signed_ext", "signed_log"] if args.map_type == "all" else [args.map_type]
    methods = ["nearest", "interpolated"] if args.method == "both" else [args.method]
    model_label = _model_name(reader, model_path)
    if args.auto_max_range:
        print(f"Auto max_range derived from data: {max_range:.4f}")

    for map_type in map_types:
        maps = ensure_multiplication_maps(map_sizes, map_type=map_type, max_range=max_range)
        for map_size in map_sizes:
            uint8_map = maps[map_size]
            for method in methods:
                print(f"\nEvaluating tensor '{args.tensor}' slice {weight_slice.shape} | map={map_type} size={map_size} method={method} max_range={max_range}, baseline={baseline_label}")
                rows = evaluate_tensor_rows(
                    weight_slice,
                    input_vec,
                    uint8_map,
                    map_type=map_type,
                    method=method,
                    float_range=float_range,
                    baseline_dtype=baseline_dtype,
                )
                summary = summarize(rows)

                for r in rows:
                    print(
                        f"Row {r['row']}: ref={r['ref']:.6f}, mapped={r['mapped']:.6f}, "
                        f"abs_err={r['abs_error']:.6f}, perc_err={r['percent_error']:.4f}%, "
                        f"prod_sats={r['product_saturations']}, sum_sats={r['sum_saturations']}"
                    )

                if summary:
                    print("\nSummary:")
                    for k, v in summary.items():
                        print(f"  {k}: {v}")

                if args.plot:
                    value_bits = _value_bits_for_size(map_size)
                    for r in rows:
                        tensor_label = _sanitize(args.tensor or "tensor")
                        filename = f"{value_bits}bit/matmul_plot_{tensor_label}_row{r['row']}_{map_type}_{map_size}_{method}.png"
                        legend_labels = {
                            "float": f"Baseline {baseline_label}",
                            "mapped": f"Int lookup {map_type} {map_size}x{map_size} ({method}) ~{value_bits}-bit grid",
                        }
                        plot_title = f"{model_label} | {args.tensor} row {r['row']} | {map_type} {map_size}x{map_size} ({method}) vs {baseline_label}"
                        lookup_label = legend_labels["mapped"]
                        plotChain(
                            r["chain"],
                            filename=filename,
                            float_range=float_range,
                            title=plot_title,
                            legend_labels=legend_labels,
                            baseline_label=legend_labels["float"],
                            lookup_label=lookup_label,
                            value_bits=value_bits,
                            show_error=args.plot_show_error,
                            ylabel="Accumulated sum",
                        )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
