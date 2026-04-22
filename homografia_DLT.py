import numpy as np

def homografia_dlt(pts_3D, pts_2D):
    n_pts = len(pts_3D)
    A =np.zeros((2*n_pts,9))
    for i in range(n_pts):
        X, Y = pts_3D[i][0], pts_3D[i][1]
        u, v = pts_2D[i][0], pts_2D[i][1]
        A[2 * i] = [-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, v * X, v * Y, v]
    U, S, Vt = np.linalg.svd(A)
    vector_h = Vt[-1, :]
    H = vector_h.reshape((3, 3))
    H = H / H[2, 2]
    return H
