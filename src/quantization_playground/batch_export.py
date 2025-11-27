"""
Run multiple simulations and export results into examples/simulation_runs/<steps>_steps/.

Usage:
    python -m quantization_playground.batch_export
"""
import argparse
from pathlib import Path

from quantization_playground.export_examples import export_simulation_examples
from quantization_playground.paths import PROJECT_ROOT
from quantization_playground.simulation import BASELINE_CHOICES, run_simulation


def parse_steps(steps_arg: str) -> list[int]:
    return [int(s) for s in steps_arg.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run simulations for multiple step counts and export examples.")
    parser.add_argument("--steps", type=str, default="100,1000", help="Comma-separated list of chain lengths to run (e.g., 100,1000).")
    parser.add_argument("--max-range", type=float, default=2.0, help="Max magnitude of representable float range.")
    parser.add_argument("--baseline-dtype", type=str, default="float16", choices=BASELINE_CHOICES, help="Reference precision (float8 allowed; falls back to float16 if unsupported).")
    parser.add_argument("--show-error", action="store_true", help="Include percent error subplot in saved figures.")
    parser.add_argument("--dest-root", type=Path, default=PROJECT_ROOT / "examples" / "simulation_runs", help="Root folder for exports.")
    args = parser.parse_args()

    steps_list = parse_steps(args.steps)
    for steps in steps_list:
        print(f"\n--- Running simulation for {steps} steps ---")
        run_simulation(max_range=args.max_range, steps=steps, baseline_dtype_name=args.baseline_dtype, show_error=args.show_error)

        dest = Path(args.dest_root) / f"{steps}_steps"
        export_simulation_examples(dest=dest, force=True)
        print(f"Exported plots to {dest}")


if __name__ == "__main__":
    main()
