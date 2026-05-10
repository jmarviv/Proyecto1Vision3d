import cv2
import numpy as np
import time
import ZHANG as Z

#Este código es el encargado de realizar las imágenes para
#pasarselas a la función zhang, también devuelve la matriz K al MAIN o la imprime por pantalla
def calibracion():
    diccionario_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parametros_aruco = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(diccionario_aruco, parametros_aruco)
    fotos_necesarias = 40
    lista_fotos = []

    medidas_reales = np.array([
        # --- ARUCO ID 0 (Arriba-Izquierda en tu pared) ---
        [0.0000, 0.3700], [0.0825, 0.3700], [0.0825, 0.2875], [0.0000, 0.2875],

        # --- ARUCO ID 1 (Abajo-Izquierda en tu pared) ---
        [0.0000, 0.0825], [0.0825, 0.0825], [0.0825, 0.0000], [0.0000, 0.0000],

        # --- ARUCO ID 2 (Arriba-Derecha en tu pared) ---
        [0.2875, 0.3700], [0.3700, 0.3700], [0.3700, 0.2875], [0.2875, 0.2875],

        # --- ARUCO ID 3 (Abajo-Derecha en tu pared) ---
        [0.2875, 0.0825], [0.3700, 0.0825], [0.3700, 0.0000], [0.2875, 0.0000]
    ], dtype=np.float32)

    url_camara = "http://10.68.175.34:8080/video"
    cap = cv2.VideoCapture(url_camara)

    tiempo_ultimo_disparo = time.time()
    intervalo_segundos = 0.6

    print("\n--- MODO CAPTURA AUTOMÁTICA (0.6s) ACTIVADO ---")
    print("Mueve el móvil SUAVEMENTE. El programa hará las fotos solo cuando vea los 4 ArUcos.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("La cámara no va")
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esquinas, ids, _ = detector.detectMarkers(gris)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, esquinas, ids)

        cv2.imshow("Calibracion Automatica - Mueve la camara despacio", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if ids is not None and len(ids) == 4:

            tiempo_actual = time.time()
            if (tiempo_actual - tiempo_ultimo_disparo) >= intervalo_segundos:
                indices_ordenados = np.argsort(ids.flatten())
                puntos_2D_esta_foto = []

                for i in indices_ordenados:
                    for pixel in esquinas[i][0]:
                        puntos_2D_esta_foto.append(pixel)

                lista_fotos.append(np.array(puntos_2D_esta_foto))
                print(f"📸 ¡Click automático! Foto {len(lista_fotos)}/{fotos_necesarias} guardada.")

                tiempo_ultimo_disparo = tiempo_actual
                if len(lista_fotos) >= fotos_necesarias:
                    break

        # SI PULSAS LA 'Q' -> Salimos de emergencia
        if tecla == ord('q') or tecla == ord('Q'):
            print("\nSaliendo antes de tiempo...")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nIniciando los cálculos de Zhang. El SVD está trabajando...")

    if len(lista_fotos) >= 3:
        Matriz_K = Z.ZHANG(lista_fotos, medidas_reales)
        print("\n--- MATRIZ INTRÍNSECA (K) OBTENIDA ---")
        print(np.round(Matriz_K, 2))
        return Matriz_K
    else:
        print("\n⚠️ No conseguiste suficientes fotos. Necesitas al menos 3 para que las matemáticas funcionen.")
        return False


