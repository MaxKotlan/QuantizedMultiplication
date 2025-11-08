import numpy as np
from PIL import Image
import os

def load_multiplication_map(size, folder='./multiplication_maps'):
    path = os.path.join(folder, f'signed_{size}x{size}.png')
    img = Image.open(path)
    arr_uint8 = np.array(img, dtype=np.uint8)
    return arr_uint8


uint8_map = load_multiplication_map(16)
print(uint8_map.shape, uint8_map.dtype)
print(uint8_map)

