import numpy as np
def calcular_distancia_manual(H, K):
    K_inv = np.linalg.inv(K)
    #h2 no lo usamos, pero podríamos haber cambiado h1 por h2
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    r1_prima = np.dot(K_inv, h1)

    escala = 1.0 / np.linalg.norm(r1_prima)

    t_prima = np.dot(K_inv, h3)
    vector_t = escala * t_prima


    return vector_t