"""
Helper to copy the latest simulation outputs into the examples folder.

Usage:
    python -m quantization_playground.export_examples
"""
from pathlib import Path
import argparse
import shutil

from .paths import PROJECT_ROOT, SIMULATION_DIR


def export_simulation_examples(
    source: Path = SIMULATION_DIR,
    dest: Path = PROJECT_ROOT / "examples" / "simulation_runs",
    force: bool = True,
) -> None:
    source = Path(source)
    dest = Path(dest)

    if not source.exists() or not any(source.iterdir()):
        raise FileNotFoundError(f"No simulation outputs found in {source}")

    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source, dest, dirs_exist_ok=not force)
    print(f"Copied simulation outputs from {source} -> {dest}")


def main():
    parser = argparse.ArgumentParser(description="Copy simulation outputs into examples/simulation_runs.")
    parser.add_argument("--source", type=Path, default=SIMULATION_DIR, help="Source simulation directory.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=PROJECT_ROOT / "examples" / "simulation_runs",
        help="Destination folder (defaults to examples/simulation_runs).",
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Do not delete the destination before copying (default is to clobber).",
    )
    args = parser.parse_args()
    export_simulation_examples(source=args.source, dest=args.dest, force=not args.no_force)


if __name__ == "__main__":
    main()
