import cv2
import numpy as np
import calibracion as C
import homografia_DLT as h
import Calcular_Distancia as D


#APUNTAR AQUÍ LA URL DEL SERVIDOR DE LA CÁMARA
url_camara = "http://10.68.175.34:8080/video"

#APUNTAR AQUI LAS MEDIDAS DE LOS ARUCOS
medidas_reales = np.array([
    # --- ARUCO ID 0  ---
    [0.0000, 0.3700], [0.0825, 0.3700], [0.0825, 0.2875], [0.0000, 0.2875],
    # --- ARUCO ID 1  ---
    [0.0000, 0.0825], [0.0825, 0.0825], [0.0825, 0.0000], [0.0000, 0.0000],
    # --- ARUCO ID 2  ---
    [0.2875, 0.3700], [0.3700, 0.3700], [0.3700, 0.2875], [0.2875, 0.2875],
    # --- ARUCO ID 3  ---
    [0.2875, 0.0825], [0.3700, 0.0825], [0.3700, 0.0000], [0.2875, 0.0000]
], dtype=np.float32)


def main():
    K = C.calibracion()
    print("✅ Calibración hecha y Matriz K obtenida")
    diccionario_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parametros = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(diccionario_aruco, parametros)


    cap = cv2.VideoCapture(url_camara)

    print("--- SISTEMA DE TELEMETRÍA ACTIVO ---")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esquinas, ids, _ = detector.detectMarkers(gris)

        if ids is not None and len(ids) == 4:
            cv2.aruco.drawDetectedMarkers(frame, esquinas, ids)

            indices_ordenados = np.argsort(ids.flatten())
            puntos_2D = []
            for i in indices_ordenados:
                for pixel in esquinas[i][0]:
                    puntos_2D.append(pixel)
            puntos_2D = np.array(puntos_2D, dtype=np.float32)
            H = h.homografia_dlt(medidas_reales, puntos_2D)


            vector_t = D.calcular_distancia_manual(H, K)

            distancia_total = np.linalg.norm(vector_t)

            cv2.putText(frame, f"Dist. TOTAL: {distancia_total:.3f} m", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Telemetria ArUco - Homografia Constante", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()