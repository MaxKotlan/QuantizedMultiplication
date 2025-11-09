import numpy as np

def float_to_log_index(f, max_abs_fz=4, size=256):
    # normalize -max_abs..max_abs -> -1..1
    norm = f / max_abs_fz
    sign = np.sign(norm)
    abs_val = np.abs(norm)
    abs_val = max(abs_val, 1e-12)
    abs_scaled = np.log1p(9 * abs_val) / np.log(10)
    fz_log = sign * abs_scaled
    idx = round((fz_log + 1) * (size-1)/2)
    return int(np.clip(idx, 0, size-1))

def log_index_to_float(idx, max_abs_fz=4, size=256):
    fz_log = idx / ((size-1)/2) - 1
    sign = np.sign(fz_log)
    abs_scaled = np.abs(fz_log)
    fz_norm = sign * (10**abs_scaled - 1)/9
    return fz_norm * max_abs_fz
