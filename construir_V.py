import numpy as np

def construir_v(col_i, col_j):
    hi1, hi2, hi3 = col_i[0], col_i[1], col_i[2]
    hj1, hj2, hj3 = col_j[0], col_j[1], col_j[2]
    v = np.array([
        hi1 * hj1,
        hi1 * hj2 + hi2 * hj1,
        hi2 * hj2,
        hi3 * hj1 + hi1 * hj3,
        hi3 * hj2 + hi2 * hj3,
        hi3 * hj3
    ])
    return v