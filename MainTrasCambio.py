import cv2
import numpy as np
import ZHANG_TrasRevision
import geometria_robusta
# URL del servidor de la cámara (igual que en main.py)
url_camara = "http://10.73.127.25:8080/video"

# Medidas reales de los marcadores ArUco (extraídas de main.py)
# Se ajustan a un plano 2D (Z=0) para la calibración con el metodo de Zhang.
medidas_reales_2D = np.array([
    # --- ARUCO ID 0 ---
    [0.0000, 0.3700], [0.0825, 0.3700], [0.0825, 0.2875], [0.0000, 0.2875],
    # --- ARUCO ID 1 ---
    [0.0000, 0.0825], [0.0825, 0.0825], [0.0825, 0.0000], [0.0000, 0.0000],
    # --- ARUCO ID 2 ---
    [0.2875, 0.3700], [0.3700, 0.3700], [0.3700, 0.2875], [0.2875, 0.2875],
    # --- ARUCO ID 3 ---
    [0.2875, 0.0825], [0.3700, 0.0825], [0.3700, 0.0000], [0.2875, 0.0000]
], dtype=np.float32)

# Convertimos las medidas 2D a 3D (añadiendo Z=0) para que sean compatibles con
# la función de homografía si esta lo requiere.
medidas_reales_3D = np.hstack([medidas_reales_2D, np.zeros((medidas_reales_2D.shape[0], 1), dtype=np.float32)])


def main():
    # Configuración del detector de ArUco (igual que en main.py)
    diccionario_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parametros = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(diccionario_aruco, parametros)

    cap = cv2.VideoCapture(url_camara)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara en la URL: {url_camara}")
        return

    lista_fotos = []
    num_fotos_capturadas = 0
    num_fotos_deseadas = 40

    print("--- INICIO DEL PROCESO DE CALIBRACIÓN MANUAL ---")
    print(f"Se deben capturar {num_fotos_deseadas} imágenes.")
    print("Instrucciones:")
    print("1. Muestre el patrón de ArUcos a la cámara desde diferentes ángulos y distancias.")
    print("2. Asegúrese de que los 4 marcadores son visibles.")
    print("3. Pulse la tecla 'ESPACIO' para capturar una imagen.")
    print("4. Pulse la tecla 'q' para salir del programa.")

    while num_fotos_capturadas < num_fotos_deseadas:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo recibir el frame de la cámara.")
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esquinas, ids, _ = detector.detectMarkers(gris)

        display_frame = frame.copy()
        if ids is not None and len(ids) == 4:
            cv2.aruco.drawDetectedMarkers(display_frame, esquinas, ids)
            # Mensaje para indicar que se pueden capturar los puntos
            cv2.putText(display_frame, "Listo para capturar. Pulse ESPACIO", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar contador
        texto_contador = f"Fotos: {num_fotos_capturadas}/{num_fotos_deseadas}"
        cv2.putText(display_frame, texto_contador, (display_frame.shape[1] - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Calibracion Manual - Pulse ESPACIO para capturar", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Calibración cancelada por el usuario.")
            break
        elif key == ord(' '):  # Barra espaciadora
            if ids is not None and len(ids) == 4:
                # Ordenar los puntos según los IDs de los ArUcos
                indices_ordenados = np.argsort(ids.flatten())
                puntos_2D_capturados = []
                for i in indices_ordenados:
                    for pixel in esquinas[i][0]:
                        puntos_2D_capturados.append(pixel)

                puntos_2D_capturados = np.array(puntos_2D_capturados, dtype=np.float32)
                lista_fotos.append(puntos_2D_capturados)
                num_fotos_capturadas += 1
                print(f"¡Foto {num_fotos_capturadas} de {num_fotos_deseadas} capturada!")
            else:
                print("Error en la captura: No se detectaron los 4 marcadores. Inténtelo de nuevo.")



    cap.release()
    cv2.destroyAllWindows()

    if len(lista_fotos) == num_fotos_deseadas:
        print("\n--- Iniciando cálculo de la matriz de calibración con el método de Zhang ---")
        # Pasamos las medidas reales en 3D (con Z=0) a la función de Zhang
        K = ZHANG_TrasRevision.ZHANG(lista_fotos, medidas_reales_3D)
        print("\n✅ Calibración completada.")
        print("Matriz de calibración K obtenida:")
        print(K)
        print("\n--- Iniciando Fase 2: Visualización de Ejes 3D ---")
        print("Mueve la cámara. Pulsa 'q' para salir.")

        # Volvemos a abrir la cámara para el modo en vivo
        cap_live = cv2.VideoCapture(url_camara)
        historial_homografias = []


        while True:
            ret, frame = cap_live.read()
            if not ret: break

            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            esquinas, ids, _ = detector.detectMarkers(gris)

            display_frame = frame.copy()

            # Si vemos los 4 ArUcos, hacemos la magia
            if ids is not None and len(ids) == 4:
                # 1. El pacto de sangre: ordenar las esquinas
                indices_ordenados = np.argsort(ids.flatten())
                puntos_2D_vivos = []
                for i in indices_ordenados:
                    for pixel in esquinas[i][0]:
                        puntos_2D_vivos.append(pixel)

                puntos_2D_vivos = np.array(puntos_2D_vivos, dtype=np.float32)

                # 2. Calcular la Homografía robusta para ESTE frame
                H_vivo = geometria_robusta.homografia_ransac_normalizada(medidas_reales_3D, puntos_2D_vivos)

                # 3. Dibujar los ejes XYZ proyectados
                if H_vivo is not None:
                    historial_homografias.append(H_vivo)
                    display_frame = geometria_robusta.dibujar_ejes_3d(display_frame, K, H_vivo, longitud_eje=0.15)

            cv2.imshow("Fase 2 - Realidad Aumentada 3D", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if 'H_vivo' in locals() and H_vivo is not None:
                    print("\n[INFO] Generando análisis 3D del frame actual...")
                    geometria_robusta.mostrar_frame_actual_3d(K, H_vivo, medidas_reales_3D)
                break



        cap_live.release()
        cv2.destroyAllWindows()
        # =================================================================

        # Al salir del bucle, generamos el gráfico pro:
        if len(historial_homografias) > 0:
            print("\nGenerando mapa 3D de Orientación Exterior...")
            geometria_robusta.mostrar_trayectoria_3d(K, historial_homografias, medidas_reales_3D)
        else:
            print("\nNo se capturaron suficientes imágenes para realizar la calibración.")


if __name__ == "__main__":
    main()
