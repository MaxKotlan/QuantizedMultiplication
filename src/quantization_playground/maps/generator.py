from pathlib import Path
import numpy as np
from PIL import Image

from ..paths import MAPS_DIR

MAP_SIZES = (4, 8, 16, 32, 64, 128, 256, 512)


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _bit_depth(size: int) -> int:
    return int(np.ceil(np.log2(size))) if size > 0 else 0


def _size_subdir(output_dir: Path, size: int) -> Path:
    bit_dir = output_dir / f"{_bit_depth(size)}bit"
    bit_dir.mkdir(parents=True, exist_ok=True)
    return bit_dir


def _save_map(arr: np.ndarray, target_dir: Path, filename: str) -> None:
    """Save the raw map; if 16-bit, also save an 8-bit preview for visibility."""
    target_dir = Path(target_dir)
    path = target_dir / filename
    Image.fromarray(arr).save(path)
    if arr.dtype == np.uint16:
        max_val = int(arr.max()) if arr.size else 1
        if max_val <= 0:
            max_val = 1
        preview = np.clip(np.round(arr.astype(np.float32) / max_val * 255), 0, 255).astype(np.uint8)
        preview_name = Path(filename).stem + "_preview.png"
        Image.fromarray(preview).save(target_dir / preview_name)


def _scale_to_dtype(val: float, size: int, dtype) -> int:
    """
    Scale a raw map value (0..size-1) into the dtype range.
    For uint8 we keep 0..size-1. For uint16 (used for size>256) we spread to 0..65535.
    """
    if dtype == np.uint8:
        return int(np.clip(round(val), 0, size - 1))
    # uint16 path: spread raw 0..(size-1) to full uint16 range
    scale = 65535 / (size - 1)
    return int(np.clip(round(val * scale), 0, 65535))


def generateUnsigned(suffix: str = "", output_dir: Path = MAPS_DIR) -> None:
    for size in MAP_SIZES:
        target_dir = _size_subdir(output_dir, size)
        dtype = np.uint16 if size > 256 else np.uint8
        gspace = np.zeros((size, size), dtype=dtype)
        max_val = size - 1
        for x in range(size):
            for y in range(size):
                q = x / size
                p = y / size
                r = q * p
                g_raw = r * max_val
                gspace[x, y] = _scale_to_dtype(g_raw, size, dtype)

        _save_map(gspace.astype(dtype), target_dir, f"unsigned_{size}x{size}{suffix}.png")


def generateSigned(suffix: str = "", output_dir: Path = MAPS_DIR) -> None:
    for size in MAP_SIZES:
        target_dir = _size_subdir(output_dir, size)
        dtype = np.uint16 if size > 256 else np.uint8
        gspace = np.zeros((size, size), dtype=dtype)
        for x in range(size):
            for y in range(size):
                fx = (x - (size - 1) / 2) / ((size - 1) / 2)
                fy = (y - (size - 1) / 2) / ((size - 1) / 2)
                fz = fx * fy
                g_raw = (fz + 1) * 127.5  # 0..255 when size=256
                gspace[x, y] = _scale_to_dtype(g_raw, size, dtype)

        _save_map(gspace, target_dir, f"signed_{size}x{size}{suffix}.png")


def generateSignedExtended(max_range: float = 2.0, suffix: str = "", output_dir: Path = MAPS_DIR) -> None:
    for size in MAP_SIZES:
        target_dir = _size_subdir(output_dir, size)
        half = (size - 1) / 2
        max_val = size - 1
        half_range = max_val / 2
        dtype = np.uint16 if size > 256 else np.uint8
        gspace = np.zeros((size, size), dtype=dtype)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * max_range
                fy = (y - half) / half * max_range
                fz = fx * fy
                fz_clamped = np.clip(fz, -max_range, max_range)
                g_raw = (fz_clamped / max_range) * half_range + half_range
                gspace[x, y] = _scale_to_dtype(g_raw, size, dtype)

        _save_map(gspace, target_dir, f"signed_extended_{size}x{size}{suffix}.png")


def generateSignedExtendedWarped(max_range: float = 2.0, suffix: str = "", output_dir: Path = MAPS_DIR) -> None:
    for size in MAP_SIZES:
        target_dir = _size_subdir(output_dir, size)
        half = (size - 1) / 2
        max_val = size - 1
        dtype = np.uint16 if size > 256 else np.uint8
        gspace = np.zeros((size, size), dtype=dtype)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * max_range
                fy = (y - half) / half * max_range
                fz = fx * fy

                # asinh behaves ~linear near 0 but compresses extremes
                k = 20.0  # larger = more zoom near zero
                max_prod = max_range ** 2
                warped = np.sign(fz) * np.arcsinh(k * abs(fz)) / np.arcsinh(k * max_prod)
                g_raw = (warped * 0.5 + 0.5) * max_val
                gspace[x, y] = _scale_to_dtype(g_raw, size, dtype)

        _save_map(gspace, target_dir, f"signed_extended_warped_{size}x{size}{suffix}.png")


def generateSignedExtendedLog(max_range: float = 2.0, suffix: str = "", output_dir: Path = MAPS_DIR) -> None:
    for size in MAP_SIZES:
        target_dir = _size_subdir(output_dir, size)
        half = (size - 1) / 2
        max_val = size - 1
        half_range = max_val / 2
        dtype = np.uint16 if size > 256 else np.uint8
        gspace = np.zeros((size, size), dtype=dtype)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * max_range
                fy = (y - half) / half * max_range
                fz = fx * fy

                fz_clamped = np.clip(fz, -max_range, max_range)
                fz_norm = fz_clamped / max_range
                sign = np.sign(fz_norm)
                abs_scaled = np.log1p(9 * np.abs(fz_norm)) / np.log(10)
                fz_log = sign * abs_scaled
                g_raw = (fz_log + 1) * half_range
                gspace[x, y] = _scale_to_dtype(g_raw, size, dtype)

        _save_map(gspace, target_dir, f"signed_log_{size}x{size}{suffix}.png")


def main(max_range: float | None = None, suffix: str | None = None, output_dir: Path | None = None) -> None:
    # Allow programmatic call with parameters; CLI falls back to argparse
    if max_range is None or suffix is None:
        import argparse

        parser = argparse.ArgumentParser(description="Generate multiplication maps.")
        parser.add_argument("--max-range", type=float, default=2.0, help="Max magnitude of representable float range (symmetric ±max_range).")
        parser.add_argument("--suffix", type=str, default="", help="Optional filename suffix for distinguishing ranges.")
        parser.add_argument("--output-dir", type=str, default=None, help="Directory to write maps into (defaults to data/multiplication_maps).")
        args = parser.parse_args()
        max_range = float(args.max_range)
        suffix = args.suffix
        output_dir = Path(args.output_dir) if args.output_dir else MAPS_DIR

    output_dir = _prepare_output_dir(output_dir or MAPS_DIR)

    generateUnsigned(suffix=suffix, output_dir=output_dir)
    generateSigned(suffix=suffix, output_dir=output_dir)
    generateSignedExtended(max_range=max_range, suffix=suffix, output_dir=output_dir)
    generateSignedExtendedLog(max_range=max_range, suffix=suffix, output_dir=output_dir)
    generateSignedExtendedWarped(max_range=max_range, suffix=suffix, output_dir=output_dir)


if __name__ == "__main__":
    main()
