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
                fx = (x - half) / half
                fy = (y - half) / half
                fz = fx * fy
                gspace[x, y] = round((fz + 1) * 127.5)

        Image.fromarray(gspace).save(f'./multiplication_maps/signed_{size}x{size}.png')



generateUnsigned()
generateSigned()

