import numpy as np
import construir_V as CONSV
import geometria_robusta as GEO_ROB

def ZHANG(lista_fotos, medidas_reales):
    lista_Homografias_hechas= []
    for puntos_utiles_piexeles in lista_fotos:
        H = GEO_ROB.homografia_ransac_normalizada(medidas_reales, puntos_utiles_piexeles)
        lista_Homografias_hechas.append(H)

    V = np.zeros((2 * len(lista_Homografias_hechas), 6))
    for i, H in enumerate(lista_Homografias_hechas):
        h1 = H[:,0]
        h2 = H[:,1]
        V[2*i] = CONSV.construir_v(h1,h2)
        V[2*i+1] = CONSV.construir_v(h1,h1) - CONSV.construir_v(h2,h2)

    U, S, Vt= np.linalg.svd(V)
    b = Vt[-1,:]

    B11, B12, B22, B13, B23, B33 = b
    if B11 < 0:
        B11, B12, B22, B13, B23, B33 = -B11, -B12, -B22, -B13, -B23, -B33

    y0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
    lambda_val = B33 - (B13 ** 2 + y0 * (B12 * B13 - B11 * B23)) / B11

    alpha_x = np.sqrt(lambda_val / B11)
    alpha_y = np.sqrt(lambda_val * B11 / (B11 * B22 - B12 ** 2))
    ese = -B12 * alpha_x ** 2 * alpha_y / lambda_val
    x0 = (ese * y0 / alpha_y) - (B13 * alpha_x ** 2 / lambda_val)

    K = np.array([
        [alpha_x, ese, x0],
        [0, alpha_y, y0],
        [0, 0, 1]
    ])

    return K
