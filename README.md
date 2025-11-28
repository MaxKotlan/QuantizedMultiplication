# Quantization Playground

This repo explores “integer space” multiplication via precomputed lookup tables and compares it against native floating‑point math. The goal is to understand how much error accumulates when products are approximated with low‑precision tables (e.g., uint8 grids), and whether the approach could be viable for integer‑friendly hardware (think faster integer pipelines, quantized ML, or embedded settings).

## What’s here
- `src/quantization_playground/maps/`: PNG map generator/loader (`ensure_multiplication_maps` auto-generates). Maps are grouped under `data/multiplication_maps/<bits>bit/` based on their effective value levels (e.g., `8bit/` holds the 256×256 tables).
- `src/quantization_playground/algorithms/`: nearest-neighbor lookup with optional stochastic rounding + bilinear interpolation.
- `src/quantization_playground/simulation.py`: runs long chains of multiplies, compares against float, saves plots.
- `src/quantization_playground/plotting.py`: plotting helper targeting `data/simulation/`.
- `data/`: generated artifacts (`multiplication_maps/` + `simulation/`); gitignored except for `.gitkeep`.
- `examples/simulation_runs/`: example chain plots from previous runs.
- `tests/`: quick smoke tests that exercise the lookup + interpolation paths.
- `python -m quantization_playground.export_examples`: copy the latest `data/simulation/` outputs into `examples/simulation_runs/` (clobbers existing examples).
- `python -m quantization_playground.batch_export`: run multiple simulations (default 100 and 1000 steps) and export each into `examples/simulation_runs/<steps>_steps/`.

## Why it matters
Precomputing products into small integer tables is a crude but fast way to approximate multiplication. If the accumulated error stays small, you could imagine integer‑only pipelines (e.g., quantized ML or special hardware) benefiting from the speed/size of integer math while retaining acceptable accuracy. This repo measures how quickly errors diverge across map sizes, encodings, and numeric ranges.

## Quick start
1) Install deps (uses a venv in these examples):
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

2) Run a simulation (maps are auto‑generated if missing for the given range):
```bash
MPLCONFIGDIR="$PWD/.mplconfig" . .venv/bin/activate && \
python -m quantization_playground.simulation --max-range 2.0 --steps 1024 --baseline-dtype float16
```
- Pass `--simulation-type dot` to switch from the default chained multiply walk to a multiply‑accumulate (dot product) run, which mimics matrix multiplication-style error accumulation.
- `--max-range`: symmetric range ±R encoded in the maps (default 2.0).
- `--steps`: length of the multiplication chain (default 1024).
- `--baseline-dtype`: reference precision (`float16` by default; `float32`/`float64` available; `float8` if your NumPy build supports it, otherwise it falls back to float16 with a note).

Outputs:
- Plots in `data/simulation/<bits>bit/chain_plot_<map>_<size>_<method>.png` (bits reflect map size: 4×4→2bit, …, 512×512→9bit).
- Console summary of final errors per variant.

## Working with ranges < 1
Small ranges make the tables extremely coarse. The simulation shrinks the random step jitter and enables stochastic rounding on nearest lookups when `max-range < 1`, but error still grows quickly because the grid has few effective bins. Interpolation helps but cannot fully recover precision at very low ranges.

## Map generation (manual)
Maps are normally generated on demand by the simulation. To regenerate explicitly:
```bash
. .venv/bin/activate
python -m quantization_playground.maps.generator --max-range 2.0 --output-dir data/multiplication_maps
```
Files are written to `data/multiplication_maps/` with a `_r<range>` suffix.
Maps are organized by effective value depth: `2bit/` (4×4), `3bit/` (8×8), …, `8bit/` (256×256), `9bit/` (512×512).

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
- Tweak `max-range`, `steps`, and the jitter in `src/quantization_playground/simulation.py` to explore stability.
- Add/remove encodings or change grid sizes in `src/quantization_playground/maps/generator.py`.
- Compare methods by inspecting the saved plots and console metrics.
