import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time
from ultralytics import YOLO

# --- Configuración ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Carga modelos YOLO
model_captafaros = YOLO('captaPT/best.pt').to(device)
model_tachas = YOLO('tachasPT/best.pt').to(device)
model_senaleticas = YOLO('PkPT/best.pt').to(device)

conf_threshold = 0.3

# Parámetros cámara (distorsión y matriz intrínseca)
K = np.array([[1500, 0, 1352], [0, 1500, 760], [0, 0, 1]])
D = np.array([-0.25, 0.03, 0, 0])

# ORB y Matcher
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Inicialización variables odometría
prev_gray = None
prev_kp, prev_des = None, None
R_f = np.eye(3)
t_f = np.zeros((3, 1))
posiciones_vehiculo = [np.array([0.0, 0.0])]

# Carga datos GPS
gps_data = pd.read_excel("metadata/3.2 - 01_04 Tramo B1-B2.xlsx")
gps_coords = gps_data[['Latitud', 'Longitud']].to_numpy()
gps_times = gps_data['Tiempo'].to_numpy()
gps_posiciones_vehiculo = []

# Función detección simplificada (filtrado por confianza)
def detectar_objetos(frame):
    res_c = model_captafaros(frame, device=device)[0]
    res_t = model_tachas(frame, device=device)[0]
    res_s = model_senaleticas(frame, device=device)[0]
    def filtrar(res):
        boxes = res.boxes.data.cpu().numpy()
        return boxes[boxes[:,4] > conf_threshold]
    return filtrar(res_c), filtrar(res_t), filtrar(res_s)

# Video
cap = cv2.VideoCapture('videos/3.2 - 01_04 Tramo B1-B2.MP4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('resultado_transformado.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), fps,
                      (frame_width, frame_height))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Corrección de distorsión
    frame_undist = cv2.undistort(frame, K, D)

    # Detección (opcional para tu caso, aquí solo para ejemplo)
    #captafaros, tachas, senaleticas = detectar_objetos(frame_undist)

    gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

    # Detectar y describir keypoints
    kp, des = orb.detectAndCompute(gray, None)

    if prev_gray is not None and prev_des is not None and des is not None and len(kp) > 20 and len(prev_kp) > 20:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 30:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

            # Calcular Essential Matrix con parámetros reales de cámara
            E, mask = cv2.findEssentialMat(dst_pts, src_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and E.shape == (3,3):
                _, R, t, mask_pose = cv2.recoverPose(E, dst_pts, src_pts, K)
                escala_asumida = 1.0
                # Actualizar pose global
                t_f += escala_asumida * R_f @ t
                R_f = R @ R_f
                posiciones_vehiculo.append(np.array([t_f[0][0], t_f[2][0]]))

    prev_gray, prev_kp, prev_des = gray, kp, des

    # Incrustar trayectoria en frame
    for i in range(1, len(posiciones_vehiculo)):
        x1, y1 = posiciones_vehiculo[i-1]
        x2, y2 = posiciones_vehiculo[i]
        pt1 = (int(x1*10 + frame_width//2), int(y1*10 + frame_height//2))
        pt2 = (int(x2*10 + frame_width//2), int(y2*10 + frame_height//2))
        cv2.line(frame_undist, pt1, pt2, (0,0,255), 2)

    out.write(frame_undist)
    cv2.imshow('Odometría Visual', cv2.resize(frame_undist, (0,0), fx=0.5, fy=0.5))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Graficar trayectoria
odo = np.array(posiciones_vehiculo)
plt.plot(odo[:, 0], odo[:, 1], 'g-', label='Odometría Visual')
plt.title("Trayectoria estimada por odometría visual")
plt.xlabel("X")
plt.ylabel("Z")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
