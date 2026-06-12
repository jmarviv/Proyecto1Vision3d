import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def normalizar_puntos(puntos):
    """
    Aplica la normalización isotrópica de Hartley.
     Centra los puntos en (0,0) y los escala a una distancia media de sqrt(2).
    """
    # 1. Centroide
    centroide = np.mean(puntos, axis=0)

    # 2. Desplazar al origen
    puntos_centrados = puntos - centroide

    # 3. Escalar
    distancias = np.linalg.norm(puntos_centrados, axis=1)
    distancia_media = np.mean(distancias)
    escala = np.sqrt(2) / distancia_media

    # 4. Matriz de transformación T
    T = np.array([
        [escala, 0, -escala * centroide[0]],
        [0, escala, -escala * centroide[1]],
        [0, 0, 1]
    ])

    # 5. Aplicar transformación
    puntos_homogeneos = np.column_stack((puntos, np.ones(len(puntos))))
    puntos_normalizados = np.dot(T, puntos_homogeneos.T).T[:, :2]

    return puntos_normalizados, T


def calcular_homografia_dlt(src, dst):
    """SVD puro para 4 puntos normalizados."""
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
    """
    Calcula la Homografía resistente a outliers usando Normalización + RANSAC.
    """
    num_puntos = len(puntos_img)
    mejor_H = None
    max_inliers = 0

    # Z=0 ya está en tu main, pero nos aseguramos de usar solo X e Y para el DLT 2D
    mundo_2d = puntos_mundo[:, :2]
    mundo_hom = np.column_stack((mundo_2d, np.ones(num_puntos)))

    for _ in range(iteraciones):
        # Elegir 4 índices al azar
        indices = random.sample(range(num_puntos), 4)
        src_muestra = mundo_2d[indices]
        dst_muestra = puntos_img[indices]

        # Normalizar la muestra
        src_norm, T_src = normalizar_puntos(src_muestra)
        dst_norm, T_dst = normalizar_puntos(dst_muestra)

        # DLT con puntos normalizados
        H_norm = calcular_homografia_dlt(src_norm, dst_norm)

        # Des-normalizar
        H_temp = np.linalg.inv(T_dst) @ H_norm @ T_src
        H_temp = H_temp / H_temp[2, 2]

        # Test de Inliers (Comprobar error contra todos los puntos)
        proyecciones = np.dot(H_temp, mundo_hom.T).T
        proyecciones_2d = proyecciones[:, :2] / proyecciones[:, 2:]

        errores = np.linalg.norm(puntos_img - proyecciones_2d, axis=1)
        inliers = np.sum(errores < umbral)

        if inliers > max_inliers:
            max_inliers = inliers
            mejor_H = H_temp

        if max_inliers == num_puntos:
            break  # Es perfecta

    return mejor_H


def procrustes_ortogonal(R_aprox):
    """
    Aplica SVD para encontrar la matriz de rotación pura (ortogonal)
    más cercana a la matriz aproximada. (Fusiello, Sección 5.2.1)
    """
    U, S, Vt = np.linalg.svd(R_aprox)
    R_pura = np.dot(U, Vt)

    # Si es una reflexión en lugar de rotación, la corregimos
    if np.linalg.det(R_pura) < 0:
        Vt[-1, :] *= -1
        R_pura = np.dot(U, Vt)

    return R_pura


def dibujar_ejes_3d(frame, K, H, longitud_eje=0.15):
    """
    Dibuja los ejes X(Rojo), Y(Verde), Z(Azul) en el frame de vídeo
    usando la matriz K y la Homografía del frame actual.
    longitud_eje está en metros (0.15 = 15 cm).
    """
    # 1. Extracción de R y t a partir de H y K
    K_inv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    r1_prima = np.dot(K_inv, h1)
    r2_prima = np.dot(K_inv, h2)
    t_prima = np.dot(K_inv, h3)

    # Calculamos el factor de escala (lambda)
    escala = 1.0 / np.linalg.norm(r1_prima)

    r1 = r1_prima * escala
    r2 = r2_prima * escala
    r3 = np.cross(r1, r2)

    # Aplicamos Análisis Ortogonal de Procrustes para forzar 90 grados perfectos
    R_aprox = np.column_stack((r1, r2, r3))
    R = procrustes_ortogonal(R_aprox)

    # Vector de traslación real (Mundo -> Cámara)
    t = t_prima * escala

    # 2. Definimos los puntos de los ejes en el Mundo 3D
    # (El origen 0,0,0 es la esquina del ArUco 0)
    # Z es negativo porque, por convención geométrica, la cámara mira hacia Z positivo,
    # así que para salir de la pared hacia la cámara, vamos en negativo.
    puntos_3D = np.array([
        [0, 0, 0],  # Origen
        [longitud_eje, 0, 0],
        [0, longitud_eje, 0],
        [0, 0, -longitud_eje]
    ])

    # 3. Proyección Estenopeica: x = K * [R|t] * X
    puntos_2D_pixeles = []
    for P_3D in puntos_3D:
        # Transformación Rígida (Pasamos del Mundo a la Cámara)
        P_camara = np.dot(R, P_3D) + t

        # Proyección en la lente (Pasamos de la Cámara a la Imagen/Píxeles)
        P_imagen_homogeneo = np.dot(K, P_camara)

        # Des-homogeneizamos dividiendo por Z para obtener el píxel final en la pantalla
        u = int(P_imagen_homogeneo[0] / P_imagen_homogeneo[2])
        v = int(P_imagen_homogeneo[1] / P_imagen_homogeneo[2])
        puntos_2D_pixeles.append((u, v))

    # 4. Dibujar en el Frame de OpenCV
    origen = puntos_2D_pixeles[0]
    pt_x = puntos_2D_pixeles[1]
    pt_y = puntos_2D_pixeles[2]
    pt_z = puntos_2D_pixeles[3]

    # Líneas con formato BGR de OpenCV (Grosor 4)
    cv2.line(frame, origen, pt_x, (0, 0, 255), 4)  # Rojo  (Eje X)
    cv2.line(frame, origen, pt_y, (0, 255, 0), 4)  # Verde (Eje Y)
    cv2.line(frame, origen, pt_z, (255, 0, 0), 4)  # Azul  (Eje Z)

    return frame


def mostrar_trayectoria_3d(K, historial_H, medidas_reales_3D, scale=0.04):
    """
    Genera el mapa 3D con los ArUcos en la pared, mostrando TODOS los frames
    del historial representados como vectores de orientación (ejes X, Y, Z).
    Visto desde el lado opuesto (Cámara en X negativo).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    K_inv = np.linalg.inv(K)
    trayectoria_C = []

    for H in historial_H:
        # 1. Extracción de la pose como siempre
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

        # TRANSFORMACIÓN AL LADO OPUESTO DE LA PARED (La vista que cuadramos antes)
        C_plot = [C[2], -C[0], -C[1]]
        trayectoria_C.append(C_plot)

        # 2. NUEVA PARTE: CÁLCULO Y DIBUJO DE LOS VECTORES DE ORIENTACIÓN (QUIVER)
        # Las filas de R representan los ejes de la cámara en el sistema del mundo
        # Aplicamos la misma rotación de ejes de la gráfica a cada vector: [Z, -X, -Y]
        v_x = [R[0, 2], -R[0, 0], -R[0, 1]]  # Eje X local (Rojo)
        v_y = [R[1, 2], -R[1, 0], -R[1, 1]]  # Eje Y local (Verde)
        v_z = [R[2, 2], -R[2, 0], -R[2, 1]]  # Eje Z local (Azul - Dirección de mirada)

        # Dibujamos las tres flechas naciendo en el centro óptico (C_plot)
        # El parámetro 'scale' controla la longitud de las flechitas en el gráfico
        ax.quiver(C_plot[0], C_plot[1], C_plot[2], v_x[0], v_x[1], v_x[2], length=scale, color='red', linewidth=1.2)
        ax.quiver(C_plot[0], C_plot[1], C_plot[2], v_y[0], v_y[1], v_y[2], length=scale, color='green', linewidth=1.2)
        ax.quiver(C_plot[0], C_plot[1], C_plot[2], v_z[0], v_z[1], v_z[2], length=scale, color='blue', linewidth=1.2)

    # 3. Dibujar la línea de la trayectoria que une todos los puntos
    if trayectoria_C:
        trayectoria_C = np.array(trayectoria_C)
        ax.plot(trayectoria_C[:, 0], trayectoria_C[:, 1], trayectoria_C[:, 2], color='black', linestyle='--',
                linewidth=1, label='Trayectoria')

    # --- DIBUJAR LA PARED AMARILLA REFLEJADA ---
    min_y, max_y = np.min(-medidas_reales_3D[:, 0]), np.max(-medidas_reales_3D[:, 0])
    min_z, max_z = np.min(-medidas_reales_3D[:, 1]), np.max(-medidas_reales_3D[:, 1])

    yy, zz = np.meshgrid([min_y - 0.1, max_y + 0.1], [min_z - 0.1, max_z + 0.1])
    xx = np.zeros_like(yy)

    ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.3)

    # Dibujar los puntos exactos de los ArUcos
    ax.scatter(np.zeros_like(medidas_reales_3D[:, 0]), -medidas_reales_3D[:, 0], -medidas_reales_3D[:, 1],
               color='black', s=20, label='Pared ArUco')

    ax.set_xlabel('Eje X (Profundidad Invertida)')
    ax.set_ylabel('Eje Y (Ancho Lateral)')
    ax.set_zlabel('Eje Z (Altura)')
    ax.set_title('Trayectoria Completa de la Cámara (Vectores Ortogonales)')
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.show()
def mostrar_frame_actual_3d(K, H, medidas_reales_3D, scale=0.1):
    """
    Genera un mapa 3D de UN SOLO FRAME con los ArUcos en la pared IZQUIERDA.
    """
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

    # Transformación a la pared izquierda
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

    # --- PLANO AMARILLO YZ ---
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