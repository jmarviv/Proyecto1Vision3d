import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def normalizar_puntos(puntos):
    centroide = np.mean(puntos, axis=0)
    puntos_centrados = puntos - centroide

    distancias = np.linalg.norm(puntos_centrados, axis=1)
    distancia_media = np.mean(distancias)
    escala = np.sqrt(2) / distancia_media

    T = np.array([
        [escala, 0, -escala * centroide[0]],
        [0, escala, -escala * centroide[1]],
        [0, 0, 1]
    ])

    puntos_homogeneos = np.column_stack((puntos, np.ones(len(puntos))))
    puntos_normalizados = np.dot(T, puntos_homogeneos.T).T[:, :2]

    return puntos_normalizados, T




def calcular_homografia_dlt(src, dst):
    A = []
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 3)




def homografia_ransac_normalizada(puntos_mundo, puntos_img, iteraciones=500, umbral=3.0):
    num_puntos = len(puntos_img)
    mejor_H = None
    max_inliers = 0

    mundo_2d = puntos_mundo[:, :2]
    mundo_hom = np.column_stack((mundo_2d, np.ones(num_puntos)))

    for _ in range(iteraciones):
        indices = random.sample(range(num_puntos), 4)
        src_muestra = mundo_2d[indices]
        dst_muestra = puntos_img[indices]

        src_norm, T_src = normalizar_puntos(src_muestra)
        dst_norm, T_dst = normalizar_puntos(dst_muestra)

        H_norm = calcular_homografia_dlt(src_norm, dst_norm)

        H_temp = np.linalg.inv(T_dst) @ H_norm @ T_src
        H_temp = H_temp / H_temp[2, 2]

        proyecciones = np.dot(H_temp, mundo_hom.T).T
        proyecciones_2d = proyecciones[:, :2] / proyecciones[:, 2:]

        errores = np.linalg.norm(puntos_img - proyecciones_2d, axis=1)
        inliers = np.sum(errores < umbral)

        if inliers > max_inliers:
            max_inliers = inliers
            mejor_H = H_temp

        if max_inliers == num_puntos:
            break

    return mejor_H




def procrustes_ortogonal(R_aprox):
    U, S, Vt = np.linalg.svd(R_aprox)
    R_pura = np.dot(U, Vt)

    if np.linalg.det(R_pura) < 0:
        Vt[-1, :] *= -1
        R_pura = np.dot(U, Vt)

    return R_pura




def dibujar_ejes_3d(frame, K, H, longitud_eje=0.15):
    K_inv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    r1_prima = np.dot(K_inv, h1)
    r2_prima = np.dot(K_inv, h2)
    t_prima = np.dot(K_inv, h3)

    escala = 1.0 / np.linalg.norm(r1_prima)

    r1 = r1_prima * escala
    r2 = r2_prima * escala
    r3 = np.cross(r1, r2)

    R_aprox = np.column_stack((r1, r2, r3))
    R = procrustes_ortogonal(R_aprox)

    t = t_prima * escala

    puntos_3D = np.array([
        [0, 0, 0],
        [longitud_eje, 0, 0],
        [0, longitud_eje, 0],
        [0, 0, -longitud_eje]
    ])

    puntos_2D_pixeles = []
    for P_3D in puntos_3D:
        P_camara = np.dot(R, P_3D) + t

        P_imagen_homogeneo = np.dot(K, P_camara)

        u = int(P_imagen_homogeneo[0] / P_imagen_homogeneo[2])
        v = int(P_imagen_homogeneo[1] / P_imagen_homogeneo[2])
        puntos_2D_pixeles.append((u, v))

    origen = puntos_2D_pixeles[0]
    pt_x = puntos_2D_pixeles[1]
    pt_y = puntos_2D_pixeles[2]
    pt_z = puntos_2D_pixeles[3]

    cv2.line(frame, origen, pt_x, (0, 0, 255), 4)
    cv2.line(frame, origen, pt_y, (0, 255, 0), 4)
    cv2.line(frame, origen, pt_z, (255, 0, 0), 4)

    return frame


def mostrar_trayectoria_3d(K, historial_H, medidas_reales_3D, scale=0.06):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    K_inv = np.linalg.inv(K)
    trayectoria_C = []

    for H in historial_H:
        h1 = H[:, 0];
        h2 = H[:, 1];
        h3 = H[:, 2]
        r1_prima = np.dot(K_inv, h1);
        r2_prima = np.dot(K_inv, h2);
        t_prima = np.dot(K_inv, h3)
        escala = 1.0 / np.linalg.norm(r1_prima)
        r1 = r1_prima * escala;
        r2 = r2_prima * escala;
        r3 = np.cross(r1, r2)
        R_aprox = np.column_stack((r1, r2, r3))
        R = procrustes_ortogonal(R_aprox)
        t = t_prima * escala
        C = -np.dot(R.T, t)

        C_plot = [C[2], -C[0], -C[1]]
        trayectoria_C.append(C_plot)


        dir_x = R[2, 2]
        dir_y = -R[0, 2]
        dir_z = -R[1, 2]

        ax.quiver(C_plot[0], C_plot[1], C_plot[2], dir_x, dir_y, dir_z,
                  length=scale, color='blue', linewidth=1.5, normalize=True)

    if trayectoria_C:
        trayectoria_C = np.array(trayectoria_C)
        ax.plot(trayectoria_C[:, 0], trayectoria_C[:, 1], trayectoria_C[:, 2],
                color='black', linestyle='--', linewidth=1, label='Trayectoria')

    min_y, max_y = np.min(-medidas_reales_3D[:, 0]), np.max(-medidas_reales_3D[:, 0])
    min_z, max_z = np.min(-medidas_reales_3D[:, 1]), np.max(-medidas_reales_3D[:, 1])

    yy, zz = np.meshgrid([min_y - 0.1, max_y + 0.1], [min_z - 0.1, max_z + 0.1])
    xx = np.zeros_like(yy)

    ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.3)
    ax.scatter(np.zeros_like(medidas_reales_3D[:, 0]), -medidas_reales_3D[:, 0], -medidas_reales_3D[:, 1],
               color='black', s=20, label='Pared ArUco')

    ax.set_xlabel('Eje X (Profundidad Invertida)')
    ax.set_ylabel('Eje Y (Ancho Lateral)')
    ax.set_zlabel('Eje Z (Altura)')
    ax.set_title('Trayectoria de la Cámara (Vector de Dirección)')
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.show()



def mostrar_frame_actual_3d(K, H, medidas_reales_3D, scale=0.1):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    K_inv = np.linalg.inv(K)

    h1 = H[:, 0];
    h2 = H[:, 1];
    h3 = H[:, 2]
    r1_prima = np.dot(K_inv, h1);
    r2_prima = np.dot(K_inv, h2);
    t_prima = np.dot(K_inv, h3)
    escala = 1.0 / np.linalg.norm(r1_prima)
    r1 = r1_prima * escala;
    r2 = r2_prima * escala;
    r3 = np.cross(r1, r2)
    R_aprox = np.column_stack((r1, r2, r3))
    R = procrustes_ortogonal(R_aprox)
    t = t_prima * escala
    C = -np.dot(R.T, t)

    euler_angles = cv2.RQDecomp3x3(R)[0]
    pitch, yaw, roll = euler_angles

    C_plot = [-C[2], C[0], -C[1]]

    esquinas_cam = np.array([
        [1, 0.75, 2], [-1, 0.75, 2], [-1, -0.75, 2], [1, -0.75, 2]
    ]) * scale
    esquinas_mundo = (np.dot(R.T, esquinas_cam.T)).T + C
    esquinas_plot = np.column_stack((-esquinas_mundo[:, 2], esquinas_mundo[:, 0], -esquinas_mundo[:, 1]))

    caras = [[C_plot, esquinas_plot[0], esquinas_plot[1]], [C_plot, esquinas_plot[1], esquinas_plot[2]],
             [C_plot, esquinas_plot[2], esquinas_plot[3]], [C_plot, esquinas_plot[3], esquinas_plot[0]],
             [esquinas_plot[0], esquinas_plot[1], esquinas_plot[2], esquinas_plot[3]]]
    ax.add_collection3d(Poly3DCollection(caras, facecolors='orange', linewidths=1, edgecolors='black', alpha=0.6))

    min_y, max_y = np.min(medidas_reales_3D[:, 0]), np.max(medidas_reales_3D[:, 0])
    min_z, max_z = np.min(-medidas_reales_3D[:, 1]), np.max(-medidas_reales_3D[:, 1])
    yy, zz = np.meshgrid([min_y - 0.1, max_y + 0.1], [min_z - 0.1, max_z + 0.1])
    xx = np.zeros_like(yy)
    ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.3)

    ax.scatter(np.zeros_like(medidas_reales_3D[:, 0]), medidas_reales_3D[:, 0], -medidas_reales_3D[:, 1], color='black',
               s=20, label='Pared ArUco')

    info_text = (
        f"POSICIÓN (Metros):\n"
        f"X (Profundidad a la pared): {-C[2]:.3f} m\n"
        f"Y (Desvío Lateral): {C[0]:.3f} m\n"
        f"Z (Altura respecto al ArUco): {-C[1]:.3f} m\n\n"
        f"ROTACIÓN (Grados):\n"
        f"Pitch: {pitch:.1f}º\n"
        f"Yaw: {yaw:.1f}º\n"
        f"Roll: {roll:.1f}º"
    )

    ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black'))

    ax.set_xlabel('Eje X (Profundidad)')
    ax.set_ylabel('Eje Y (Ancho Lateral)')
    ax.set_zlabel('Eje Z (Altura)')
    ax.set_title('Análisis de Orientación Exterior (Vista de Perfil)')
    ax.set_box_aspect([1, 1, 1])
    plt.show()