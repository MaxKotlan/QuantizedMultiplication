import numpy as np
from PIL import Image

map_sizes = [4, 8, 16, 32, 64, 128, 256]

for size in map_sizes:
    gspace = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            half=size//2
            q = (x/half) # modified to use 127 instead of 255
            p = (y/half) # modified to use 127 instead of 255
            r = q * p
            g = round(r * half) + half # shift to allow for negative values
            gspace[x,y] = g*(256//size)

    Image.fromarray(gspace.astype(np.uint8)).save(f'./multiplication_maps/signed_{size}x{size}.png')
