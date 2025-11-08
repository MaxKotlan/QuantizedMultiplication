import numpy as np
from PIL import Image
import os

def load_multiplication_map(size, folder='./multiplication_maps'):
    path = os.path.join(folder, f'signed_{size}x{size}.png')
    img = Image.open(path)
    arr_uint8 = np.array(img, dtype=np.uint8)
    return arr_uint8


uint8_map = load_multiplication_map(16)
# print(uint8_map.shape, uint8_map.dtype)
# print(uint8_map)


def multiplyIntSpace(a, b, uint8_map):
    size = uint8_map.shape[0]
    x = int(a * (size-1) / 255)
    y = int(b * (size-1) / 255)
    x = min(max(x, 0), size-1)
    y = min(max(y, 0), size-1)
    return uint8_map[x, y]

# this was generated and works.... but I dont understand really
# def multiplyFloatSpace(fa, fb, uint8_map):
#     size = uint8_map.shape[0]

#     fac = np.clip(fa, -1.0, 1.0)
#     fbc = np.clip(fb, -1.0, 1.0)

#     # map [-1,1] → index 0..size-1
#     ia = round((fac + 1) / 2 * (size-1))
#     ib = round((fbc + 1) / 2 * (size-1))

#     ia = min(max(ia, 0), size-1)
#     ib = min(max(ib, 0), size-1)

#     ir = uint8_map[ia, ib]

#     # map 0..255 back to -1..1
#     mapped = ir / 255 * 2 - 1
#     return mapped


def multiplyFloatSpace(fa, fb, uint8_map):
    fac = np.clip(fa, -1.0, 1.0)
    fbc = np.clip(fb, -1.0, 1.0)

    ia = round((fac + 1) * 127.5)
    ib = round((fbc + 1) * 127.5)
    ir = multiplyIntSpace(ia, ib, uint8_map)
    return (ir - 127.5) / 127.5

def testFloat(fa, fb, uint8_map):
    regular = fa * fb
    mapped_value = multiplyFloatSpace(fa, fb, uint8_map)
    error_percent = abs(mapped_value - regular) * 100  # treat max possible value as 1
    return fa, fb, regular, mapped_value, error_percent

def printTestFloatResults(results):
    fa, fb, regular, mapped_value, error_percent = results
    print(f"Regular multiplication: {fa} * {fb} = {regular:.6f}")
    print(f"Mapped multiplication: uint8={mapped_value}")
    print(f"Error (positive %): {error_percent:.2f}%")

results = testFloat(.5, .1, uint8_map)
printTestFloatResults(results)
