import numpy as np
from PIL import Image
import os


def generateUnsigned(suffix=""):
    map_sizes = [4, 8, 16, 32, 64, 128, 256]

    for size in map_sizes:
        gspace = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                q = (x/size)
                p = (y/size)
                r = q * p
                g = round(r * size)
                gspace[x,y] = g*(256//size)

        Image.fromarray(gspace.astype(np.uint8)).save(f'./multiplication_maps/unsigned_{size}x{size}{suffix}.png')


def generateSigned(suffix=""):
    map_sizes = [4, 8, 16, 32, 64, 128, 256]

    for size in map_sizes:
        half = size // 2
        gspace = np.zeros((size, size), dtype=np.uint8)

        for x in range(size):
            for y in range(size):
                # fx = (x - half) / half
                # fy = (y - half) / half
                # fz = fx * fy
                fx = (x - (size-1)/2) / ((size-1)/2)
                fy = (y - (size-1)/2) / ((size-1)/2)
                fz = fx * fy

                gspace[x, y] = round((fz + 1) * 127.5)

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_{size}x{size}{suffix}.png')

def generateSignedExtended(max_range=2.0, suffix=""):
    map_sizes = [4, 8, 16, 32, 64, 128, 256]

    for size in map_sizes:
        half = (size - 1) / 2
        max_val = size - 1
        half_range = max_val / 2
        gspace = np.zeros((size, size), dtype=np.uint8)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * max_range   # -max_range..max_range
                fy = (y - half) / half * max_range   # -max_range..max_range
                fz = fx * fy                          # -(max_range^2)..(max_range^2)
                fz_clamped = np.clip(fz, -max_range, max_range)

                # Map -2..2 -> 0..255 signed style
                # Formula: gspace = round((fz / 2) * half_range + half_range)
                # Now fz=-max_range -> 0, fz=0 -> half_range, fz=max_range -> max_val
                gspace[x, y] = np.clip(round((fz_clamped / max_range) * half_range + half_range), 0, max_val)

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_extended_{size}x{size}{suffix}.png')

import numpy as np
from PIL import Image

def generateSignedExtendedWarped(max_range=2.0, suffix=""):
    map_sizes = [4, 8, 16, 32, 64, 128, 256]

    for size in map_sizes:
        half = (size - 1) / 2
        max_val = size - 1
        gspace = np.zeros((size, size), dtype=np.uint8)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * max_range   # -max_range..max_range
                fy = (y - half) / half * max_range   # -max_range..max_range
                fz = fx * fy                         # -(max_range^2)..(max_range^2)

                # --- Nonlinear warping for more detail near zero ---
                # asinh behaves ~linear near 0 but compresses extremes
                k = 20.0  # larger = more zoom near zero
                max_prod = max_range ** 2
                warped = np.sign(fz) * np.arcsinh(k * abs(fz)) / np.arcsinh(k * max_prod)

                # Map warped -1..1 to 0..255
                gspace[x, y] = np.clip(round((warped * 0.5 + 0.5) * max_val), 0, max_val)

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_extended_warped_{size}x{size}{suffix}.png')


def generateSignedExtendedLog(max_range=2.0, suffix=""):
    map_sizes = [4, 8, 16, 32, 64, 128, 256]

    for size in map_sizes:
        half = (size - 1) / 2
        max_val = size - 1
        half_range = max_val / 2
        gspace = np.zeros((size, size), dtype=np.uint8)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * max_range   # -max_range..max_range
                fy = (y - half) / half * max_range   # -max_range..max_range
                fz = fx * fy                         # -(max_range^2)..(max_range^2)

                # normalize
                fz_clamped = np.clip(fz, -max_range, max_range)
                fz_norm = fz_clamped / max_range     # -1..1

                # Signed log-like scaling for more resolution near zero
                sign = np.sign(fz_norm)
                abs_scaled = np.log1p(9 * np.abs(fz_norm)) / np.log(10)  # maps 0..1 -> 0..1
                fz_log = sign * abs_scaled

                # Map to 0..255
                gspace[x, y] = np.clip(round((fz_log + 1) * half_range), 0, max_val)

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_log_{size}x{size}{suffix}.png')


def main(max_range=None, suffix=None):
    # Allow programmatic call with parameters; CLI falls back to argparse
    if max_range is None or suffix is None:
        import argparse
        parser = argparse.ArgumentParser(description="Generate multiplication maps.")
        parser.add_argument("--max-range", type=float, default=2.0, help="Max magnitude of representable float range (symmetric ±max_range).")
        parser.add_argument("--suffix", type=str, default="", help="Optional filename suffix for distinguishing ranges.")
        args = parser.parse_args()
        max_range = float(args.max_range)
        suffix = args.suffix

    # Ensure output folder exists
    os.makedirs("./multiplication_maps", exist_ok=True)

    generateUnsigned(suffix=suffix)
    generateSigned(suffix=suffix)
    generateSignedExtended(max_range=max_range, suffix=suffix)
    generateSignedExtendedLog(max_range=max_range, suffix=suffix)
    generateSignedExtendedWarped(max_range=max_range, suffix=suffix)


if __name__ == "__main__":
    main()
