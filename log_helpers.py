import numpy as np

def float_to_log_index(f, min_f=-2, max_f=2, size=256):
    # normalize -2..2 -> -1..1
    norm = (2 * (f - min_f) / (max_f - min_f)) - 1
    sign = np.sign(norm)
    abs_scaled = np.log1p(9 * np.abs(norm)) / np.log(10)
    fz_log = sign * abs_scaled
    # map -1..1 -> 0..(size-1)
    idx = np.clip(round((fz_log + 1) * (size-1)/2), 0, size-1)
    return int(idx)

def log_index_to_float(idx, size=256, min_f=-2, max_f=2):
    # map 0..(size-1) -> -1..1
    fz_log = (idx / ((size-1)/2)) - 1
    sign = np.sign(fz_log)
    abs_scaled = np.abs(fz_log)
    norm = sign * (10**abs_scaled - 1) / 9
    # map -1..1 -> -2..2
    f = ((norm + 1)/2) * (max_f - min_f) + min_f
    return f
