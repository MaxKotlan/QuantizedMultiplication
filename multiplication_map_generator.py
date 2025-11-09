import numpy as np
from PIL import Image


def generateUnsigned():
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

        Image.fromarray(gspace.astype(np.uint8)).save(f'./multiplication_maps/unsigned_{size}x{size}.png')


def generateSigned():
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

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_{size}x{size}.png')

def generateSignedExtended():
    map_sizes = [4, 8, 16, 32, 64, 128, 256]

    for size in map_sizes:
        half = (size - 1) / 2
        gspace = np.zeros((size, size), dtype=np.uint8)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * 2   # -2..2
                fy = (y - half) / half * 2   # -2..2
                fz = fx * fy                  # -4..4

                # Map -2..2 -> 0..255 signed style
                # Formula: gspace = round((fz / 2) * 127.5 + 127.5)
                # Now fz=-2 -> 0, fz=0 -> 127.5, fz=2 -> 255
                gspace[x, y] = np.clip(round((fz / 2) * 127.5 + 127.5), 0, 255)

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_extended_{size}x{size}.png')

import numpy as np
from PIL import Image

def generateSignedExtendedWarped():
    map_sizes = [4, 8, 16, 32, 64, 128, 256]

    for size in map_sizes:
        half = (size - 1) / 2
        gspace = np.zeros((size, size), dtype=np.uint8)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * 2   # -2..2
                fy = (y - half) / half * 2   # -2..2
                fz = fx * fy                 # -4..4

                # --- Nonlinear warping for more detail near zero ---
                # asinh behaves ~linear near 0 but compresses extremes
                k = 20.0  # larger = more zoom near zero
                warped = np.sign(fz) * np.arcsinh(k * abs(fz)) / np.arcsinh(k * 4)

                # Map warped -1..1 to 0..255
                gspace[x, y] = round((warped * 0.5 + 0.5) * 255)

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_extended_warped_{size}x{size}.png')


def generateSignedExtendedLog():
    map_sizes = [4, 8, 16, 32, 64, 128, 256]

    for size in map_sizes:
        half = (size - 1) / 2
        gspace = np.zeros((size, size), dtype=np.uint8)

        for x in range(size):
            for y in range(size):
                fx = (x - half) / half * 2   # -2..2
                fy = (y - half) / half * 2   # -2..2
                fz = fx * fy                 # -4..4

                # normalize
                fz_norm = fz / 4             # -1..1

                # signed log-like scaling
                sign = np.sign(fz_norm)
                abs_scaled = np.log1p(9 * np.abs(fz_norm)) / np.log(10)  # 0..1
                fz_log = sign * abs_scaled

                # map to 0..255
                gspace[x, y] = np.clip(round((fz_log + 1) * 127.5), 0, 255)

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_log_{size}x{size}.png')


generateUnsigned()
generateSigned()
generateSignedExtended()
generateSignedExtendedLog()
generateSignedExtendedWarped()

