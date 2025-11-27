# Quantization Playground

This repo explores “integer space” multiplication via precomputed lookup tables and compares it against native floating‑point math. The goal is to understand how much error accumulates when products are approximated with low‑precision tables (e.g., uint8 grids), and whether the approach could be viable for integer‑friendly hardware (think faster integer pipelines, quantized ML, or embedded settings).

## What’s here
- `multiplication_map_generator.py`: builds multiplication maps (PNG grids) for several encodings (`signed_ext`, `signed_log`, etc.) at multiple resolutions. Maps can target different symmetric ranges via `--max-range` (default ±2). Filenames are suffixed with the range to avoid stale reuse.
- `multiplication_map_loader.py`: loads maps and will auto‑generate them on demand (`ensure_multiplication_maps`).
- `algorithms/nearest_neighbor.py`: table lookup using nearest index, with optional stochastic rounding to reduce quantization bias.
- `algorithms/bilinear.py`: bilinear interpolation over the table.
- `multiplication_compounding_simulation.py`: runs long chains of multiplies in both float and quantized space, tracks divergence, and saves plots.
- `graph.py`: plotting helper; axes scale to the chosen numeric range.
- `simulation/`: output plots from runs.

## Why it matters
Precomputing products into small integer tables is a crude but fast way to approximate multiplication. If the accumulated error stays small, you could imagine integer‑only pipelines (e.g., quantized ML or special hardware) benefiting from the speed/size of integer math while retaining acceptable accuracy. This repo measures how quickly errors diverge across map sizes, encodings, and numeric ranges.

## Quick start
1) Install deps (uses a venv in these examples):
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip numpy pillow matplotlib
```

2) Run a simulation (maps are auto‑generated if missing for the given range):
```bash
MPLCONFIGDIR="$PWD/.mplconfig" . .venv/bin/activate && \
python multiplication_compounding_simulation.py --max-range 2.0 --steps 1024
```
- `--max-range`: symmetric range ±R encoded in the maps (default 2.0).
- `--steps`: length of the multiplication chain (default 1024).

Outputs:
- Plots in `simulation/chain_plot_<map>_<size>_<method>.png`.
- Console summary of final errors per variant.

## Working with ranges < 1
Small ranges make the tables extremely coarse. The simulation shrinks the random step jitter and enables stochastic rounding on nearest lookups when `max-range < 1`, but error still grows quickly because the grid has few effective bins. Interpolation helps but cannot fully recover precision at very low ranges.

## Map generation (manual)
Maps are normally generated on demand by the simulation. To regenerate explicitly:
```bash
. .venv/bin/activate
python multiplication_map_generator.py --max-range 2.0
```
Files are written to `multiplication_maps/` with a `_r<range>` suffix.

## Interpreting results
- **Map size**: Larger grids (e.g., 256×256) track floats better; tiny grids diverge fast.
- **Encoding**: `signed_ext` is linear; `signed_log` gives more resolution near zero but can distort elsewhere.
- **Nearest vs interpolated**: Interpolation reduces quantization artifacts at the cost of some compute.
- **Stochastic rounding**: For small ranges, randomizing the index rounding can reduce bias but doesn’t eliminate coarse resolution limits.

## Caveats
- This is a toy exploration, not a production quantization pipeline.
- The lookup tables saturate at the chosen `max-range`; products outside are clamped.
- Error compounding is sensitive to the input sequence; a different seed can change trajectories.

## Extend/inspect
- Tweak `max-range`, `steps`, and the jitter in `multiplication_compounding_simulation.py` to explore stability.
- Add/remove encodings or change grid sizes in `multiplication_map_generator.py`.
- Compare methods by inspecting the saved plots and console metrics.

