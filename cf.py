# Mejora de odometría visual con acumulación de rotación y translación coherente
# Código realizado por Felipe Pereira Alarcón
# MODIFICADO: Sistema de seguimiento con Filtro de Kalman y Máquina de Estados
# MODIFICADO: Detección de 'Tacha Faltante' en tiempo real.
# MODIFICADO: Cálculo de distancia entre tachas confirmadas en Log Excel.
# MODIFICADO (por IA): Mejoras en minimapa con ruta GPS.
# MODIFICADO (por Instrucciones): Añadido seguimiento y análisis para 'captafaros'.

import cv2
import numpy as np
import torch
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import os

# --- Global parameters for Kalman Filter noise covariance multipliers ---
PROCESS_NOISE_MULTIPLIER = 0.3 # For display box 'a'
MEASUREMENT_NOISE_MULTIPLIER = 0.8 # For display box 'b'

# --- Constantes para conversión GPS ---
R_EARTH_METERS = 6371000.0
METERS_PER_DEGREE_LAT = R_EARTH_METERS * np.pi / 180.0
METERS_PER_DEGREE_LON_AT_ORIGIN = None # Se calculará con el primer punto GPS

# --- Escala para la ruta GPS en el minimapa (¡AJUSTA ESTE VALOR!) ---
# Ejemplo: 0.5 significa que 1 metro se representa como 0.5 píxeles.
# Si el vehículo se mueve 100m, ocupará 50 píxeles.
ESCALA_MAPA_GPS = 0.2 # Prueba inicial, ajusta según sea necesario

# --- INICIO: Nueva Clase para el Seguimiento de Tachas ---
class TachaTracker:
    """
    Esta clase representa una única tacha que se está siguiendo.
    Contiene el Filtro de Kalman, su estado, ID y otros atributos para gestionar su ciclo de vida.
    """
    def __init__(self, tacha_id, centro_img, centro_transformado, frame_idx):
        self.id = tacha_id
        self.display_id = None
        self.kf = cv2.KalmanFilter(4, 2) # Estado: [x, y, vx, vy], Medida: [x, y]

        # --- Configuración del Filtro de Kalman ---
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE_MULTIPLIER
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE_MULTIPLIER
        self.kf.statePost = np.array([centro_img[0], centro_img[1], 0, 0], dtype=np.float32)

        # --- Máquina de Estados y Atributos de Ciclo de Vida ---
        self.estado = 'Tentativo'
        self.centro_img = centro_img
        self.centro_transformado = centro_transformado
        self.frame_idx = frame_idx
        self.hits = 1
        self.misses = 0
        self.age = 0

    def predict(self):
        self.age += 1
        self.misses += 1
        return self.kf.predict()

    def update(self, centro_img, centro_transformado):
        self.misses = 0
        self.hits += 1
        self.centro_img = centro_img
        self.centro_transformado = centro_transformado
        medida = np.array([centro_img[0], centro_img[1]], dtype=np.float32)
        self.kf.correct(medida)

# --- FIN: Nueva Clase para el Seguimiento de Tachas ---

# --- INICIO: Nueva Clase para el Seguimiento de Captafaros --- [cite: 9, 43]
class CaptafaroTracker:
    """
    Esta clase representa un único captafaro que se está siguiendo.
    Contiene el Filtro de Kalman, su estado, ID y otros atributos para gestionar su ciclo de vida.
    """
    def __init__(self, captafaro_id, centro_img, centro_transformado, frame_idx): # [cite: 43]
        self.id = captafaro_id # [cite: 43]
        self.display_id = None # [cite: 43]
        self.kf = cv2.KalmanFilter(4, 2) # Estado: [x, y, vx, vy], Medida: [x, y] [cite: 43]

        # --- Configuración del Filtro de Kalman ---
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], # [cite: 44]
                                              [0, 1, 0, 1], # [cite: 44]
                                              [0, 0, 1, 0], # [cite: 45]
                                              [0, 0, 0, 1]], np.float32) # [cite: 45]
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], # [cite: 45]
                                               [0, 1, 0, 0]], np.float32) # [cite: 46]
        # Puedes ajustar estos multiplicadores si el ruido de los captafaros es diferente al de las tachas [cite: 46]
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE_MULTIPLIER # [cite: 46]
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE_MULTIPLIER # [cite: 46]
        self.kf.statePost = np.array([centro_img[0], centro_img[1], 0, 0], dtype=np.float32) # [cite: 47]

        # --- Máquina de Estados y Atributos de Ciclo de Vida ---
        self.estado = 'Tentativo' # [cite: 47]
        self.centro_img = centro_img # [cite: 47]
        self.centro_transformado = centro_transformado # [cite: 47]
        self.frame_idx = frame_idx # [cite: 47]
        self.hits = 1 # [cite: 47]
        self.misses = 0 # [cite: 47]
        self.age = 0 # [cite: 47]

    def predict(self): # [cite: 48]
        self.age += 1 # [cite: 48]
        self.misses += 1 # [cite: 48]
        return self.kf.predict() # [cite: 48]

    def update(self, centro_img, centro_transformado): # [cite: 48]
        self.misses = 0 # [cite: 48]
        self.hits += 1 # [cite: 48]
        self.centro_img = centro_img # [cite: 48]
        self.centro_transformado = centro_transformado # [cite: 48]
        medida = np.array([centro_img[0], centro_img[1]], dtype=np.float32) # [cite: 48]
        self.kf.correct(medida) # [cite: 48]
# --- FIN: Nueva Clase para el Seguimiento de Captafaros --- [cite: 49]


# Configuración
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print("Usando dispositivo:", device)

model_captafaros = YOLO('captaPT/best.pt').to(device)
model_tachas = YOLO('tachasPT/best.pt').to(device)
model_senaleticas = YOLO('PkPT/best.pt').to(device)

# Parámetros
conf_threshold = 0.3
MIN_TACHA_WIDTH_PX = 5
MAX_TACHA_WIDTH_PX = 80
MIN_TACHA_HEIGHT_PX = 5
MAX_TACHA_HEIGHT_PX = 80
MIN_TACHA_ASPECT_RATIO = 0.3
MAX_TACHA_ASPECT_RATIO = 2.5

# Parámetros para Captafaros (ajusta según tus datos) [cite: 49]
MIN_CAPTAFARO_WIDTH_PX = 8 # Ejemplo [cite: 49]
MAX_CAPTAFARO_WIDTH_PX = 100 # Ejemplo [cite: 49]
MIN_CAPTAFARO_HEIGHT_PX = 8 # Ejemplo [cite: 49]
MAX_CAPTAFARO_HEIGHT_PX = 100 # Ejemplo [cite: 49]
MIN_CAPTAFARO_ASPECT_RATIO = 0.5 # Ejemplo [cite: 49]
MAX_CAPTAFARO_ASPECT_RATIO = 2.0 # Ejemplo [cite: 49]


MIN_HITS_TO_CONFIRM = 4.5
MAX_MISSES_TO_DELETE = 4.5
REASSOC_DIST_THRESHOLD_MULTIPLIER = 1.5
PROXIMITY_THRESHOLD_FOR_ID_SUPPRESSION = 50
NMS_PRETRACK_DIST_THRESHOLD = 30

min_frames_entre_detecciones = 5
ventana_inicial = 5
tolerancia_frames = 2.0

K = np.array([[1500, 0, 1352], [0, 1500, 760], [0, 0, 1]])
D = np.array([-0.25, 0.03, 0, 0])

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
prev_gray, prev_kp, prev_des = None, None, None
R_f = np.eye(3)
t_f = np.zeros((3, 1))
posiciones_vehiculo = [np.array([0.0, 0.0])] # Para odometría visual

# Variables para ruta GPS
gps_data = None
gps_trayectoria = None
gps_tiempos = None
gps_posiciones_vehiculo_raw = [] # Almacena [Lat, Lon] crudas del archivo
gps_origin_lat_lon = None
gps_path_map_points = [] # Almacena puntos (x,y) para dibujar en minimapa

try:
    gps_data = pd.read_excel("metadata/3.2 - 01_04 Tramo B1-B2.xlsx")
    gps_trayectoria = gps_data[['Latitud', 'Longitud']].values
    gps_tiempos = gps_data['Tiempo'].values
except FileNotFoundError:
    print("Archivo GPS 'metadata/3.2 - 01_04 Tramo B1-B2.xlsx' no encontrado.")
    # Mantener gps_data, gps_trayectoria, gps_tiempos como None


# Funciones auxiliares
def transformar_punto(punto, matriz):
    punto_homog = np.array([[punto[0], punto[1], 1.0]]).T
    transformado = np.dot(matriz, punto_homog)
    if transformado[2] == 0:
        return np.array([float('inf'), float('inf')])
    transformado /= transformado[2]
    return transformado[0:2].flatten()

def detectar_objetos(frame):
    resultados_c = model_captafaros(frame, device=device)
    resultados_t = model_tachas(frame, device=device)
    resultados_s = model_senaleticas(frame, device=device)
    return (resultados_c[0].boxes.data.cpu().numpy(),
            resultados_t[0].boxes.data.cpu().numpy(),
            resultados_s[0].boxes.data.cpu().numpy())

def filtrar_detecciones(objetos, otros1, otros2, x0, x1, y0, y1, current_frame_idx): # For Tachas
    filtrados_detailed = []
    for obj in objetos:
        x1b, y1b, x2b, y2b, conf, cls_id = obj
        if conf < conf_threshold: continue
        cx, cy = (x1b + x2b) / 2, (y1b + y2b) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1): continue
        width_b = x2b - x1b
        height_b = y2b - y1b
        if not (MIN_TACHA_WIDTH_PX <= width_b <= MAX_TACHA_WIDTH_PX): continue
        if not (MIN_TACHA_HEIGHT_PX <= height_b <= MAX_TACHA_HEIGHT_PX): continue
        if height_b == 0: continue
        aspect_ratio_b = width_b / height_b
        if not (MIN_TACHA_ASPECT_RATIO <= aspect_ratio_b <= MAX_TACHA_ASPECT_RATIO): continue
        demasiado_cerca_otros = False
        for x1o, y1o, x2o, y2o, *_ in otros1:
            cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2
            if np.linalg.norm([cx - cxo, cy - cyo]) < 30:
                demasiado_cerca_otros = True; break
        if demasiado_cerca_otros: continue
        for x1o, y1o, x2o, y2o, *_ in otros2:
            cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2
            if np.linalg.norm([cx - cxo, cy - cyo]) < 30:
                demasiado_cerca_otros = True; break
        if demasiado_cerca_otros: continue
        filtrados_detailed.append({'cx': cx, 'cy': cy, 'frame_idx': current_frame_idx})
    return filtrados_detailed

def filtrar_captafaros(objetos, otros1, otros2, x0, x1, y0, y1, current_frame_idx): # [cite: 51]
    filtrados_detailed = [] # [cite: 51]
    for obj in objetos: # [cite: 51]
        x1b, y1b, x2b, y2b, conf, cls_id = obj # [cite: 51]
        if conf < conf_threshold: continue # [cite: 51]
        cx, cy = (x1b + x2b) / 2, (y1b + y2b) / 2 # [cite: 51]
        if not (x0 <= cx <= x1 and y0 <= cy <= y1): continue # [cite: 51]
        width_b = x2b - x1b # [cite: 51]
        height_b = y2b - y1b # [cite: 52]
        if not (MIN_CAPTAFARO_WIDTH_PX <= width_b <= MAX_CAPTAFARO_WIDTH_PX): continue # [cite: 52]
        if not (MIN_CAPTAFARO_HEIGHT_PX <= height_b <= MAX_CAPTAFARO_HEIGHT_PX): continue # [cite: 52]
        if height_b == 0: continue # [cite: 52]
        aspect_ratio_b = width_b / height_b # [cite: 52]
        if not (MIN_CAPTAFARO_ASPECT_RATIO <= aspect_ratio_b <= MAX_CAPTAFARO_ASPECT_RATIO): continue # [cite: 52]
        demasiado_cerca_otros = False # [cite: 52]
        # Prevent captafaros detections from being too close to tachas or senaleticas [cite: 53]
        for x1o, y1o, x2o, y2o, *_ in otros1: # assuming otros1 is tachas_raw [cite: 53]
            cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2 # [cite: 53]
            if np.linalg.norm([cx - cxo, cy - cyo]) < 30: # Use a suitable proximity threshold [cite: 53]
                demasiado_cerca_otros = True; break # [cite: 53]
        if demasiado_cerca_otros: continue # [cite: 54]
        for x1o, y1o, x2o, y2o, *_ in otros2: # assuming otros2 is senaleticas_raw [cite: 54]
            cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2 # [cite: 54]
            if np.linalg.norm([cx - cxo, cy - cyo]) < 30: # Use a suitable proximity threshold [cite: 54]
                demasiado_cerca_otros = True; break # [cite: 54]
        if demasiado_cerca_otros: continue # [cite: 55]
        filtrados_detailed.append({'cx': cx, 'cy': cy, 'frame_idx': current_frame_idx}) # [cite: 55]
    return filtrados_detailed # [cite: 55]


def detectar_tachas_faltantes(tachas_ordenadas, media_espaciado_bev, umbral_multiplicador=1.75):
    faltantes = []
    if not tachas_ordenadas or media_espaciado_bev <= 0: return faltantes
    umbral_distancia_max = media_espaciado_bev * umbral_multiplicador
    for i in range(len(tachas_ordenadas) - 1):
        tacha_actual = tachas_ordenadas[i]
        tacha_siguiente_real = tachas_ordenadas[i+1]
        pos_actual_bev = tacha_actual['centro_transformado']
        pos_siguiente_real_bev = tacha_siguiente_real['centro_transformado']
        distancia_a_siguiente_real = np.linalg.norm(pos_siguiente_real_bev - pos_actual_bev)
        if distancia_a_siguiente_real > umbral_distancia_max:
            num_faltantes_estimadas = int(round(distancia_a_siguiente_real / media_espaciado_bev)) - 1
            if num_faltantes_estimadas > 0:
                vector_direccion = (pos_siguiente_real_bev - pos_actual_bev) / (num_faltantes_estimadas + 1)
                for j in range(1, num_faltantes_estimadas + 1):
                    pos_faltante_estimada_bev = pos_actual_bev + j * vector_direccion
                    faltantes.append({
                        'tacha_id_secuencial': f"Estimada_Faltante_{tacha_actual.get('tacha_id_secuencial', tacha_actual.get('id'))}_{j}",
                        'centro_transformado': pos_faltante_estimada_bev,
                        'frame': tacha_actual['frame'], # Debería ser frame de tacha_actual
                        'clase': 'tacha_faltante_estimada',
                        'Estado': 'Faltante'
                    })
    return faltantes

# And a similar function for missing captafaros [cite: 55]
def detectar_captafaros_faltantes(captafaros_ordenadas, media_espaciado_bev, umbral_multiplicador=1.75): # [cite: 55]
    faltantes = [] # [cite: 55]
    if not captafaros_ordenadas or media_espaciado_bev <= 0: return faltantes # [cite: 55]
    umbral_distancia_max = media_espaciado_bev * umbral_multiplicador # [cite: 55]
    for i in range(len(captafaros_ordenadas) - 1): # [cite: 55]
        captafaro_actual = captafaros_ordenadas[i] # [cite: 55]
        captafaro_siguiente_real = captafaros_ordenadas[i+1] # [cite: 55]
        pos_actual_bev = captafaro_actual['centro_transformado'] # [cite: 56]
        pos_siguiente_real_bev = captafaro_siguiente_real['centro_transformado'] # [cite: 56]
        distancia_a_siguiente_real = np.linalg.norm(pos_siguiente_real_bev - pos_actual_bev) # [cite: 56]
        if distancia_a_siguiente_real > umbral_distancia_max: # [cite: 56]
            num_faltantes_estimadas = int(round(distancia_a_siguiente_real / media_espaciado_bev)) - 1 # [cite: 56]
            if num_faltantes_estimadas > 0: # [cite: 56]
                vector_direccion = (pos_siguiente_real_bev - pos_actual_bev) / (num_faltantes_estimadas + 1) # [cite: 56]
                for j in range(1, num_faltantes_estimadas + 1): # [cite: 57]
                    pos_faltante_estimada_bev = pos_actual_bev + j * vector_direccion # [cite: 57]
                    faltantes.append({ # [cite: 57]
                        'captafaro_id_secuencial': f"Estimado_Faltante_{captafaro_actual.get('captafaro_id_secuencial', captafaro_actual.get('id'))}_{j}", # [cite: 57]
                        'centro_transformado': pos_faltante_estimada_bev, # [cite: 58]
                        'frame': captafaro_actual['frame'], # [cite: 58]
                        'clase': 'captafaro_faltante_estimado', # [cite: 58]
                        'Estado': 'Faltante' # [cite: 58]
                    }) # [cite: 59]
    return faltantes # [cite: 59]


# Video
ruta_video = 'videos/3.2 - 01_04 Tramo B1-B2.MP4'
output_video_filename = '3.2-01_04 Tramo B1-B2_resultadoTransformado_Tachav3_GPSmap_Captafaros.MP4' # Nombre modificado
cap = cv2.VideoCapture(ruta_video)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el video {ruta_video}"); exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))

zona_x_inicio = int(frame_width * 0.20)
zona_x_fin = int(frame_width * 0.50)
zona_y_inicio = int(frame_height * 0.55)
zona_y_fin = frame_height
pts_origen = np.float32([[zona_x_inicio, zona_y_inicio], [zona_x_fin, zona_y_inicio], [0, zona_y_fin], [frame_width, zona_y_fin]]) #Puntos ROI Originales
ancho_transformado, alto_transformado = 400, 600
pts_destino = np.float32([[0, 0], [ancho_transformado, 0], [0, alto_transformado], [ancho_transformado, alto_transformado]])
M = cv2.getPerspectiveTransform(pts_origen, pts_destino)
M_inv = cv2.getPerspectiveTransform(pts_destino, pts_origen)

frame_idx = 0
# Tacha variables
active_trackers = []
next_tacha_id = 0
next_sequential_confirmed_id = 1
historial_confirmadas = []
ultimas_5_ids_confirmadas = []
MAX_DISPLAY_IDS = 5 # Shared for now
detecciones_frames = []
promedio_frames = 0.0
desviacion_frames = 0.0

# Captafaro variables [cite: 60]
active_captafaro_trackers = [] # [cite: 60]
next_captafaro_id = 0 # [cite: 60]
next_sequential_confirmed_captafaro_id = 1 # [cite: 60]
historial_confirmadas_captafaros = [] # [cite: 60]
ultimas_5_ids_confirmadas_captafaros = [] # [cite: 60]
detecciones_frames_captafaros = [] # [cite: 60]
promedio_frames_captafaros = 0.0 # [cite: 60]
desviacion_frames_captafaros = 0.0 # [cite: 60]

script_name = os.path.basename(__file__) if '__file__' in globals() else "beta6_GPS_Captafaros.py" # Nombre actualizado

# --- Variables para el minimapa ---
mini_mapa_h, mini_mapa_w = 200, 200
centro_mapa_x, centro_mapa_y = mini_mapa_w // 2, mini_mapa_h // 2
escala_mapa_vo = 5 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    tiempo_actual = frame_idx / fps if fps > 0 else 0
    frame_corregido = cv2.undistort(frame, K, D)

    # --- Procesamiento de Coordenadas GPS para Minimapa ---
    if gps_data is not None and gps_tiempos is not None and gps_trayectoria is not None:
        idx_gps = (np.abs(gps_tiempos - tiempo_actual)).argmin()
        if idx_gps < len(gps_trayectoria):
            gps_posiciones_vehiculo_raw.append(gps_trayectoria[idx_gps])
            current_lat, current_lon = gps_trayectoria[idx_gps]
            if gps_origin_lat_lon is None: 
                gps_origin_lat_lon = (current_lat, current_lon)
                METERS_PER_DEGREE_LON_AT_ORIGIN = METERS_PER_DEGREE_LAT * np.cos(np.radians(gps_origin_lat_lon[0]))
                gps_path_map_points.append((centro_mapa_x, centro_mapa_y))
            else:
                if METERS_PER_DEGREE_LON_AT_ORIGIN is not None:
                    delta_lat = current_lat - gps_origin_lat_lon[0]
                    delta_lon = current_lon - gps_origin_lat_lon[1]
                    dx_meters = delta_lon * METERS_PER_DEGREE_LON_AT_ORIGIN
                    dy_meters = delta_lat * METERS_PER_DEGREE_LAT 
                    map_x = centro_mapa_x + int(dx_meters * ESCALA_MAPA_GPS)
                    map_y = centro_mapa_y + int(dy_meters * ESCALA_MAPA_GPS)
                    if not gps_path_map_points or gps_path_map_points[-1] != (map_x, map_y):
                        gps_path_map_points.append((map_x, map_y))

    captafaros_raw, tachas_raw, senaleticas_raw = detectar_objetos(frame_corregido) # [cite: 17]

    # --- INICIO DEL FLUJO DE TRACKING PARA TACHAS ---
    detecciones_actuales = filtrar_detecciones(
        tachas_raw, captafaros_raw, senaleticas_raw,
        zona_x_inicio, zona_x_fin, zona_y_inicio, zona_y_fin, frame_idx
    )

    if detecciones_actuales:
        final_detections_after_nms = []
        suppressed_indices = set()
        for i in range(len(detecciones_actuales)):
            if i in suppressed_indices: continue
            current_detection = detecciones_actuales[i]
            is_current_suppressed = False # Renamed from is_current_suppressed_t
            for j in range(i + 1, len(detecciones_actuales)):
                if j in suppressed_indices: continue
                dist = np.linalg.norm([current_detection['cx'] - detecciones_actuales[j]['cx'],
                                    current_detection['cy'] - detecciones_actuales[j]['cy']])
                if dist < NMS_PRETRACK_DIST_THRESHOLD:
                    suppressed_indices.add(j)
            if not is_current_suppressed: # Renamed
                final_detections_after_nms.append(current_detection)
        detecciones_actuales = final_detections_after_nms
    
    for tracker in active_trackers: tracker.predict()

    unmatched_detections = list(detecciones_actuales)
    successfully_updated_tracker_row_indices = []
    matched_detection_indices_in_actuales = []

    if active_trackers and detecciones_actuales:
        cost_matrix = np.zeros((len(active_trackers), len(detecciones_actuales)))
        for t, tracker in enumerate(active_trackers):
            pred_x, pred_y = tracker.kf.statePre[0], tracker.kf.statePre[1]
            for d, det in enumerate(detecciones_actuales):
                det_x, det_y = det['cx'], det['cy']
                cost_matrix[t, d] = np.linalg.norm([pred_x - det_x, pred_y - det_y])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        max_dist_threshold = 40 
        temp_matched_indices = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_dist_threshold:
                tracker = active_trackers[r]
                det = detecciones_actuales[c]
                centro_img_det = np.array([det['cx'], det['cy']])
                centro_transformado_det = transformar_punto(centro_img_det, M)
                tracker.update(centro_img_det, centro_transformado_det)
                temp_matched_indices.append(c)
                successfully_updated_tracker_row_indices.append(r)
        matched_detection_indices_in_actuales = list(set(temp_matched_indices))
    unmatched_detections = [d for i, d in enumerate(detecciones_actuales) if i not in matched_detection_indices_in_actuales]

    detections_for_new_trackers = []
    if unmatched_detections:
        missed_confirmed_tracker_indices = [
            i for i, tr in enumerate(active_trackers)
            if tr.estado == 'Confirmado' and i not in successfully_updated_tracker_row_indices
        ]
        if missed_confirmed_tracker_indices and unmatched_detections:
            reassoc_cost_matrix = np.full((len(missed_confirmed_tracker_indices), len(unmatched_detections)), np.inf)
            reassoc_dist_threshold = max_dist_threshold * REASSOC_DIST_THRESHOLD_MULTIPLIER
            for i, tracker_global_idx in enumerate(missed_confirmed_tracker_indices):
                tracker = active_trackers[tracker_global_idx]
                pred_x, pred_y = tracker.kf.statePre[0], tracker.kf.statePre[1]
                for j, det in enumerate(unmatched_detections):
                    det_x, det_y = det['cx'], det['cy']
                    reassoc_cost_matrix[i, j] = np.linalg.norm([pred_x - det_x, pred_y - det_y])
            reassoc_row_ind, reassoc_col_ind = linear_sum_assignment(reassoc_cost_matrix)
            reassociated_detection_indices_in_unmatched = set()
            for r_local_idx, c_local_idx in zip(reassoc_row_ind, reassoc_col_ind):
                if reassoc_cost_matrix[r_local_idx, c_local_idx] < reassoc_dist_threshold:
                    tracker_global_idx = missed_confirmed_tracker_indices[r_local_idx]
                    tracker_to_reassociate = active_trackers[tracker_global_idx]
                    det_to_reassociate = unmatched_detections[c_local_idx]
                    centro_img_det = np.array([det_to_reassociate['cx'], det_to_reassociate['cy']])
                    centro_transformado_det = transformar_punto(centro_img_det, M)
                    tracker_to_reassociate.update(centro_img_det, centro_transformado_det)
                    reassociated_detection_indices_in_unmatched.add(c_local_idx)
            detections_for_new_trackers = [det for i, det in enumerate(unmatched_detections) if i not in reassociated_detection_indices_in_unmatched]
        else:
            detections_for_new_trackers = list(unmatched_detections)
    else:
        detections_for_new_trackers = []
 
    for det in detections_for_new_trackers:
        centro_img_new = np.array([det['cx'], det['cy']])
        centro_transformado_new = transformar_punto(centro_img_new, M)
        new_tracker = TachaTracker(next_tacha_id, centro_img_new, centro_transformado_new, frame_idx)
        active_trackers.append(new_tracker)
        next_tacha_id += 1

    newly_confirmed_this_frame_pass = []
    for tracker_candidate in active_trackers:
        if tracker_candidate.estado == 'Tentativo' and tracker_candidate.hits >= MIN_HITS_TO_CONFIRM:
            is_too_close = False
            cx_candidate, cy_candidate = tracker_candidate.centro_img[0], tracker_candidate.centro_img[1]
            for other_tracker in active_trackers:
                if other_tracker.id == tracker_candidate.id: continue
                if other_tracker.estado == 'Confirmado' and other_tracker.display_id is not None:
                    dist = np.linalg.norm([cx_candidate - other_tracker.centro_img[0], cy_candidate - other_tracker.centro_img[1]])
                    if dist < PROXIMITY_THRESHOLD_FOR_ID_SUPPRESSION:
                        is_too_close = True; break
            if is_too_close: continue
            for just_confirmed_tracker in newly_confirmed_this_frame_pass:
                dist = np.linalg.norm([cx_candidate - just_confirmed_tracker.centro_img[0], cy_candidate - just_confirmed_tracker.centro_img[1]])
                if dist < PROXIMITY_THRESHOLD_FOR_ID_SUPPRESSION:
                    is_too_close = True; break
            if not is_too_close:
                tracker_candidate.estado = 'Confirmado'
                if tracker_candidate.display_id is None:
                    tracker_candidate.display_id = next_sequential_confirmed_id
                    next_sequential_confirmed_id += 1
                    newly_confirmed_this_frame_pass.append(tracker_candidate)
                    if not detecciones_frames or (frame_idx - detecciones_frames[-1]) >= min_frames_entre_detecciones:
                        detecciones_frames.append(frame_idx)
                        if len(detecciones_frames) >= ventana_inicial + 1:
                            intervalos = [j - i for i, j in zip(detecciones_frames[:-1], detecciones_frames[1:])]
                            window_intervals = intervalos[-ventana_inicial:]
                            if window_intervals:
                                promedio_frames = sum(window_intervals) / len(window_intervals)
                                desviacion_frames = statistics.stdev(window_intervals) if len(window_intervals) > 1 else 0
                    historial_confirmadas.append({
                        'internal_id': tracker_candidate.id, 'tacha_id_secuencial': tracker_candidate.display_id,
                        'centro_transformado': tracker_candidate.centro_transformado.copy(),
                        'centro_img': tracker_candidate.centro_img.copy(),
                        'frame': tracker_candidate.frame_idx, 'clase': 'tacha'
                    })
                    new_confirmed_info_dict = {
                        "id": tracker_candidate.display_id, "cx": int(tracker_candidate.centro_img[0]),
                        "cy": int(tracker_candidate.centro_img[1])}
                    ultimas_5_ids_confirmadas.append(new_confirmed_info_dict)
                    if len(ultimas_5_ids_confirmadas) > MAX_DISPLAY_IDS: ultimas_5_ids_confirmadas.pop(0)

    final_trackers = []
    for tracker in active_trackers:
        if tracker.misses < MAX_MISSES_TO_DELETE:
            final_trackers.append(tracker)
        else:
            print(f"Eliminando tracker TACHA ID {tracker.id}, ID secuencial {tracker.display_id} por exceso de misses.")
    active_trackers = final_trackers
            
    if promedio_frames > 0 and len(detecciones_frames) > ventana_inicial:
        tiempo_desde_ultima = frame_idx - detecciones_frames[-1]
        umbral_falta = promedio_frames + tolerancia_frames * desviacion_frames
        if tiempo_desde_ultima > umbral_falta:
            cv2.putText(frame_corregido, "FALTA TACHA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    # --- FIN DEL FLUJO DE TRACKING PARA TACHAS ---


    # --- INICIO DEL NUEVO FLUJO DE TRACKING PARA CAPTAFAROS --- [cite: 61]
    # `captafaros_raw` is already returned by `detectar_objetos` [cite: 61]
    detecciones_actuales_captafaros = filtrar_captafaros( # [cite: 61]
        captafaros_raw, tachas_raw, senaleticas_raw, # Pass tachas_raw and senaleticas_raw for proximity check [cite: 61]
        zona_x_inicio, zona_x_fin, zona_y_inicio, zona_y_fin, frame_idx # [cite: 61]
    ) # [cite: 61]

    if detecciones_actuales_captafaros: # [cite: 61]
        final_detections_after_nms_captafaros = [] # [cite: 61]
        suppressed_indices_captafaros = set() # [cite: 62]
        for i in range(len(detecciones_actuales_captafaros)): # [cite: 62]
            if i in suppressed_indices_captafaros: continue # [cite: 62]
            current_detection_c = detecciones_actuales_captafaros[i] # [cite: 62]
            is_current_suppressed_c = False # [cite: 62]
            for j in range(i + 1, len(detecciones_actuales_captafaros)): # [cite: 62]
                if j in suppressed_indices_captafaros: continue # [cite: 62]
                dist_c = np.linalg.norm([current_detection_c['cx'] - detecciones_actuales_captafaros[j]['cx'], # [cite: 63]
                                    current_detection_c['cy'] - detecciones_actuales_captafaros[j]['cy']]) # [cite: 63]
                if dist_c < NMS_PRETRACK_DIST_THRESHOLD: # You might want a specific threshold for captafaros [cite: 63]
                    suppressed_indices_captafaros.add(j) # [cite: 64]
            if not is_current_suppressed_c: # [cite: 64]
                final_detections_after_nms_captafaros.append(current_detection_c) # [cite: 64]
        detecciones_actuales_captafaros = final_detections_after_nms_captafaros # [cite: 64]
    
    for tracker_c in active_captafaro_trackers: tracker_c.predict() # [cite: 64]

    unmatched_detections_captafaros = list(detecciones_actuales_captafaros) # [cite: 64]
    successfully_updated_tracker_row_indices_captafaros = [] # [cite: 64]
    matched_detection_indices_in_actuales_captafaros = [] # [cite: 64]

    if active_captafaro_trackers and detecciones_actuales_captafaros: # [cite: 64]
        cost_matrix_c = np.zeros((len(active_captafaro_trackers), len(detecciones_actuales_captafaros))) # [cite: 64]
        for t_c, tracker_c in enumerate(active_captafaro_trackers): # [cite: 64]
            pred_x_c, pred_y_c = tracker_c.kf.statePre[0], tracker_c.kf.statePre[1] # [cite: 65]
            for d_c, det_c in enumerate(detecciones_actuales_captafaros): # [cite: 65]
                det_x_c, det_y_c = det_c['cx'], det_c['cy'] # [cite: 65]
                cost_matrix_c[t_c, d_c] = np.linalg.norm([pred_x_c - det_x_c, pred_y_c - det_y_c]) # [cite: 65]
        row_ind_c, col_ind_c = linear_sum_assignment(cost_matrix_c) # [cite: 65]
        max_dist_threshold_c = 40 # Adjust if captafaros move/appear differently [cite: 65]
        temp_matched_indices_c = [] # [cite: 66]
        for r_c, c_c in zip(row_ind_c, col_ind_c): # [cite: 66]
            if cost_matrix_c[r_c, c_c] < max_dist_threshold_c: # [cite: 66]
                tracker_c = active_captafaro_trackers[r_c] # [cite: 66]
                det_c = detecciones_actuales_captafaros[c_c] # [cite: 66]
                centro_img_det_c = np.array([det_c['cx'], det_c['cy']]) # [cite: 66]
                centro_transformado_det_c = transformar_punto(centro_img_det_c, M) # [cite: 67]
                tracker_c.update(centro_img_det_c, centro_transformado_det_c) # [cite: 67]
                temp_matched_indices_c.append(c_c) # [cite: 67]
                successfully_updated_tracker_row_indices_captafaros.append(r_c) # [cite: 67]
        matched_detection_indices_in_actuales_captafaros = list(set(temp_matched_indices_c)) # [cite: 67]
    unmatched_detections_captafaros = [d for i, d in enumerate(detecciones_actuales_captafaros) if i not in matched_detection_indices_in_actuales_captafaros] # [cite: 67]

    detections_for_new_trackers_captafaros = [] # [cite: 67]
    if unmatched_detections_captafaros: # [cite: 68]
        missed_confirmed_tracker_indices_captafaros = [ # [cite: 68]
            i for i, tr in enumerate(active_captafaro_trackers) # [cite: 68]
            if tr.estado == 'Confirmado' and i not in successfully_updated_tracker_row_indices_captafaros # [cite: 68]
        ] # [cite: 68]
        if missed_confirmed_tracker_indices_captafaros and unmatched_detections_captafaros: # [cite: 68]
            reassoc_cost_matrix_c = np.full((len(missed_confirmed_tracker_indices_captafaros), len(unmatched_detections_captafaros)), np.inf) # [cite: 68]
            reassoc_dist_threshold_c = max_dist_threshold_c * REASSOC_DIST_THRESHOLD_MULTIPLIER # [cite: 68]
            for i_c, tracker_global_idx_c in enumerate(missed_confirmed_tracker_indices_captafaros): # [cite: 69]
                tracker_c = active_captafaro_trackers[tracker_global_idx_c] # [cite: 69]
                pred_x_c, pred_y_c = tracker_c.kf.statePre[0], tracker_c.kf.statePre[1] # [cite: 69]
                for j_c, det_c in enumerate(unmatched_detections_captafaros): # [cite: 69]
                    det_x_c, det_y_c = det_c['cx'], det_c['cy'] # [cite: 69]
                    reassoc_cost_matrix_c[i_c, j_c] = np.linalg.norm([pred_x_c - det_x_c, pred_y_c - det_y_c]) # [cite: 70]
            reassoc_row_ind_c, reassoc_col_ind_c = linear_sum_assignment(reassoc_cost_matrix_c) # [cite: 70]
            reassociated_detection_indices_in_unmatched_captafaros = set() # [cite: 70]
            for r_local_idx_c, c_local_idx_c in zip(reassoc_row_ind_c, reassoc_col_ind_c): # [cite: 70]
                if reassoc_cost_matrix_c[r_local_idx_c, c_local_idx_c] < reassoc_dist_threshold_c: # [cite: 70]
                    tracker_global_idx_c = missed_confirmed_tracker_indices_captafaros[r_local_idx_c] # [cite: 71]
                    tracker_to_reassociate_c = active_captafaro_trackers[tracker_global_idx_c] # [cite: 71]
                    det_to_reassociate_c = unmatched_detections_captafaros[c_local_idx_c] # [cite: 71]
                    centro_img_det_c = np.array([det_to_reassociate_c['cx'], det_to_reassociate_c['cy']]) # [cite: 71]
                    centro_transformado_det_c = transformar_punto(centro_img_det_c, M) # [cite: 72]
                    tracker_to_reassociate_c.update(centro_img_det_c, centro_transformado_det_c) # [cite: 72]
                    reassociated_detection_indices_in_unmatched_captafaros.add(c_local_idx_c) # [cite: 72]
            detections_for_new_trackers_captafaros = [det for i, det in enumerate(unmatched_detections_captafaros) if i not in reassociated_detection_indices_in_unmatched_captafaros] # [cite: 72]
        else: # [cite: 72]
            detections_for_new_trackers_captafaros = list(unmatched_detections_captafaros) # [cite: 72]
    else: # [cite: 72]
        detections_for_new_trackers_captafaros = [] # [cite: 73]
 
    for det_c in detections_for_new_trackers_captafaros: # [cite: 73]
        centro_img_new_c = np.array([det_c['cx'], det_c['cy']]) # [cite: 73]
        centro_transformado_new_c = transformar_punto(centro_img_new_c, M) # [cite: 73]
        new_tracker_c = CaptafaroTracker(next_captafaro_id, centro_img_new_c, centro_transformado_new_c, frame_idx) # [cite: 73]
        active_captafaro_trackers.append(new_tracker_c) # [cite: 73]
        next_captafaro_id += 1 # [cite: 73]

    newly_confirmed_this_frame_pass_captafaros = [] # [cite: 73]
    for tracker_candidate_c in active_captafaro_trackers: # [cite: 73]
        if tracker_candidate_c.estado == 'Tentativo' and tracker_candidate_c.hits >= MIN_HITS_TO_CONFIRM: # [cite: 73]
            is_too_close_c = False # [cite: 74]
            cx_candidate_c, cy_candidate_c = tracker_candidate_c.centro_img[0], tracker_candidate_c.centro_img[1] # [cite: 74]
            # Check proximity to other *confirmed* captafaros [cite: 74]
            for other_tracker_c in active_captafaro_trackers: # [cite: 74]
                if other_tracker_c.id == tracker_candidate_c.id: continue # [cite: 74]
                if other_tracker_c.estado == 'Confirmado' and other_tracker_c.display_id is not None: # [cite: 74]
                    dist_c = np.linalg.norm([cx_candidate_c - other_tracker_c.centro_img[0], cy_candidate_c - other_tracker_c.centro_img[1]]) # [cite: 75]
                    if dist_c < PROXIMITY_THRESHOLD_FOR_ID_SUPPRESSION: # [cite: 75]
                        is_too_close_c = True; break # [cite: 75]
            if is_too_close_c: continue # [cite: 76]
            for just_confirmed_tracker_c in newly_confirmed_this_frame_pass_captafaros: # [cite: 76]
                dist_c = np.linalg.norm([cx_candidate_c - just_confirmed_tracker_c.centro_img[0], cy_candidate_c - just_confirmed_tracker_c.centro_img[1]]) # [cite: 76]
                if dist_c < PROXIMITY_THRESHOLD_FOR_ID_SUPPRESSION: # [cite: 76]
                    is_too_close_c = True; break # [cite: 76]
            if not is_too_close_c: # [cite: 77]
                tracker_candidate_c.estado = 'Confirmado' # [cite: 77]
                if tracker_candidate_c.display_id is None: # [cite: 77]
                    tracker_candidate_c.display_id = next_sequential_confirmed_captafaro_id # [cite: 77]
                    next_sequential_confirmed_captafaro_id += 1 # [cite: 77]
                    newly_confirmed_this_frame_pass_captafaros.append(tracker_candidate_c) # [cite: 78]
                    if not detecciones_frames_captafaros or (frame_idx - detecciones_frames_captafaros[-1]) >= min_frames_entre_detecciones: # [cite: 78]
                        detecciones_frames_captafaros.append(frame_idx) # [cite: 78]
                        if len(detecciones_frames_captafaros) >= ventana_inicial + 1: # [cite: 78]
                            intervalos_c = [j - i for i, j in zip(detecciones_frames_captafaros[:-1], detecciones_frames_captafaros[1:])] # [cite: 79]
                            window_intervals_c = intervalos_c[-ventana_inicial:] # [cite: 79]
                            if window_intervals_c: # [cite: 79]
                                promedio_frames_captafaros = sum(window_intervals_c) / len(window_intervals_c) # [cite: 80]
                                desviacion_frames_captafaros = statistics.stdev(window_intervals_c) if len(window_intervals_c) > 1 else 0 # [cite: 80]
                    historial_confirmadas_captafaros.append({ # [cite: 80]
                        'internal_id': tracker_candidate_c.id, 'captafaro_id_secuencial': tracker_candidate_c.display_id, # [cite: 81]
                        'centro_transformado': tracker_candidate_c.centro_transformado.copy(), # [cite: 81]
                        'centro_img': tracker_candidate_c.centro_img.copy(), # [cite: 81]
                        'frame': tracker_candidate_c.frame_idx, 'clase': 'captafaro' # [cite: 81]
                    }) # [cite: 82]
                    new_confirmed_info_dict_c = { # [cite: 82]
                        "id": tracker_candidate_c.display_id, "cx": int(tracker_candidate_c.centro_img[0]), # [cite: 82]
                        "cy": int(tracker_candidate_c.centro_img[1])} # [cite: 82]
                    ultimas_5_ids_confirmadas_captafaros.append(new_confirmed_info_dict_c) # [cite: 83]
                    if len(ultimas_5_ids_confirmadas_captafaros) > MAX_DISPLAY_IDS: ultimas_5_ids_confirmadas_captafaros.pop(0) # [cite: 83]

    final_trackers_captafaros = [] # [cite: 83]
    for tracker_c in active_captafaro_trackers: # [cite: 83]
        if tracker_c.misses < MAX_MISSES_TO_DELETE: # [cite: 83]
            final_trackers_captafaros.append(tracker_c) # [cite: 83]
        else: # [cite: 83]
            print(f"Eliminando tracker de captafaro ID {tracker_c.id}, ID secuencial {tracker_c.display_id} por exceso de misses.") # [cite: 83]
    active_captafaro_trackers = final_trackers_captafaros # [cite: 84]
            
    if promedio_frames_captafaros > 0 and len(detecciones_frames_captafaros) > ventana_inicial: # [cite: 84]
        tiempo_desde_ultima_c = frame_idx - detecciones_frames_captafaros[-1] # [cite: 84]
        umbral_falta_c = promedio_frames_captafaros + tolerancia_frames * desviacion_frames_captafaros # [cite: 84]
        if tiempo_desde_ultima_c > umbral_falta_c: # [cite: 84]
            cv2.putText(frame_corregido, "FALTA CAPTAFARO", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA) # Different position/color [cite: 84]
    # --- FIN DEL FLUJO DE TRACKING PARA CAPTAFAROS ---


    cv2.rectangle(frame_corregido, (zona_x_inicio, zona_y_inicio), (zona_x_fin, zona_y_fin), (255, 255, 0), 2)
    # Draw Tachas
    for tracker in active_trackers:
        if tracker.estado == 'Confirmado':
            center_x_img, center_y_img = int(tracker.centro_img[0]), int(tracker.centro_img[1])
            cv2.circle(frame_corregido, (center_x_img, center_y_img), 7, (0, 255, 0), -1) # Green for tachas
            id_text_to_show = tracker.display_id if tracker.display_id is not None else tracker.id
            cv2.putText(frame_corregido, f"T_ID:{id_text_to_show}", (center_x_img + 10, center_y_img - 10), # Prefix T_ID
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Captafaros [cite: 84]
    for tracker_c in active_captafaro_trackers: # [cite: 85]
        if tracker_c.estado == 'Confirmado': # [cite: 85]
            center_x_img_c, center_y_img_c = int(tracker_c.centro_img[0]), int(tracker_c.centro_img[1]) # [cite: 85]
            cv2.circle(frame_corregido, (center_x_img_c, center_y_img_c), 7, (255, 0, 0), -1) # Blue color for captafaros [cite: 85]
            id_text_to_show_c = tracker_c.display_id if tracker_c.display_id is not None else tracker_c.id # [cite: 85]
            cv2.putText(frame_corregido, f"C_ID:{id_text_to_show_c}", (center_x_img_c + 10, center_y_img_c - 10), # Prefix C_ID [cite: 85]
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # [cite: 86]


    # --- Odometría Visual (Cálculos se mantienen para el gráfico final de matplotlib) ---
    gray = cv2.cvtColor(frame_corregido, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if prev_gray is not None and prev_des is not None and des is not None and len(kp) > 0:
        try:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 20:
                src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                E, mask_e = cv2.findEssentialMat(dst_pts, src_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R_vo, t_vo, mask_rp = cv2.recoverPose(E, dst_pts, src_pts, K) 
                    escala_asumida = 1.0 
                    t_f += escala_asumida * (R_f @ t_vo)
                    R_f = R_vo @ R_f
                    posiciones_vehiculo.append(np.array([t_f[0][0], t_f[2][0]]))
        except cv2.error as e:
            pass 
    prev_gray, prev_kp, prev_des = gray, kp, des


    # --- Dibujo del Mini-mapa MEJORADO CON GPS ---
    mini_mapa_bg_color = (220, 220, 220)
    mini_mapa = np.full((mini_mapa_h, mini_mapa_w, 3), mini_mapa_bg_color, dtype=np.uint8)
    grid_color = (180, 180, 180); axes_color = (0, 0, 0); text_color = (0,0,0)
    
    grid_spacing_pixels = 25 
    ref_scale_for_grid_text = ESCALA_MAPA_GPS if ESCALA_MAPA_GPS > 0 else escala_mapa_vo 
    if ref_scale_for_grid_text <= 0: ref_scale_for_grid_text = 1.0

    for x_g in range(0, mini_mapa_w, grid_spacing_pixels): cv2.line(mini_mapa, (x_g, 0), (x_g, mini_mapa_h), grid_color, 1)
    for y_g in range(0, mini_mapa_h, grid_spacing_pixels): cv2.line(mini_mapa, (0, y_g), (mini_mapa_w, y_g), grid_color, 1)
    cv2.line(mini_mapa, (0, centro_mapa_y), (mini_mapa_w, centro_mapa_y), axes_color, 1)
    cv2.line(mini_mapa, (centro_mapa_x, 0), (centro_mapa_x, mini_mapa_h), axes_color, 1)

    scale_font_scale = 0.3; scale_font_thickness = 1
    for x_pix in range(centro_mapa_x, mini_mapa_w, grid_spacing_pixels): 
        dist_m = (x_pix - centro_mapa_x) / ref_scale_for_grid_text
        if dist_m > 0:
            cv2.line(mini_mapa, (x_pix, centro_mapa_y - 3), (x_pix, centro_mapa_y + 3), axes_color, 1)
            cv2.putText(mini_mapa, f"{dist_m:.0f}m", (x_pix + 2, centro_mapa_y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale_font_scale, text_color, scale_font_thickness)
    for x_pix in range(centro_mapa_x - grid_spacing_pixels, 0, -grid_spacing_pixels): 
        dist_m = (x_pix - centro_mapa_x) / ref_scale_for_grid_text
        cv2.line(mini_mapa, (x_pix, centro_mapa_y - 3), (x_pix, centro_mapa_y + 3), axes_color, 1)
        cv2.putText(mini_mapa, f"{dist_m:.0f}m", (x_pix + 2, centro_mapa_y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale_font_scale, text_color, scale_font_thickness)
    for y_pix in range(centro_mapa_y + grid_spacing_pixels, mini_mapa_h, grid_spacing_pixels):
        dist_m = (y_pix - centro_mapa_y) / ref_scale_for_grid_text
        cv2.line(mini_mapa, (centro_mapa_x - 3, y_pix), (centro_mapa_x + 3, y_pix), axes_color, 1)
        cv2.putText(mini_mapa, f"{dist_m:.0f}m", (centro_mapa_x + 5, y_pix + 3), cv2.FONT_HERSHEY_SIMPLEX, scale_font_scale, text_color, scale_font_thickness)
    for y_pix in range(centro_mapa_y - grid_spacing_pixels, 0, -grid_spacing_pixels):
        dist_m = (y_pix - centro_mapa_y) / ref_scale_for_grid_text
        cv2.line(mini_mapa, (centro_mapa_x - 3, y_pix), (centro_mapa_x + 3, y_pix), axes_color, 1)
        cv2.putText(mini_mapa, f"{dist_m:.0f}m", (centro_mapa_x + 5, y_pix + 3), cv2.FONT_HERSHEY_SIMPLEX, scale_font_scale, text_color, scale_font_thickness)
    cv2.putText(mini_mapa, "0", (centro_mapa_x + 2, centro_mapa_y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale_font_scale, text_color, scale_font_thickness)

    path_color_gps = (255, 0, 255); current_pos_color_gps = (0, 255, 255); arrow_color_gps = (255, 165, 0)
    if len(gps_path_map_points) > 1:
        for i in range(1, len(gps_path_map_points)):
            p1_mapa_gps, p2_mapa_gps = gps_path_map_points[i-1], gps_path_map_points[i]
            if (0 <= p1_mapa_gps[0] < mini_mapa_w and 0 <= p1_mapa_gps[1] < mini_mapa_h and
                0 <= p2_mapa_gps[0] < mini_mapa_w and 0 <= p2_mapa_gps[1] < mini_mapa_h):
                cv2.line(mini_mapa, p1_mapa_gps, p2_mapa_gps, path_color_gps, 2)
    if gps_path_map_points:
        current_pos_on_map_gps = gps_path_map_points[-1]
        if (0 <= current_pos_on_map_gps[0] < mini_mapa_w and 0 <= current_pos_on_map_gps[1] < mini_mapa_h):
            cv2.circle(mini_mapa, current_pos_on_map_gps, 4, current_pos_color_gps, -1)
        if len(gps_path_map_points) >= 2:
            prev_pos_on_map_gps = gps_path_map_points[-2]
            if abs(current_pos_on_map_gps[0] - prev_pos_on_map_gps[0]) > 0 or abs(current_pos_on_map_gps[1] - prev_pos_on_map_gps[1]) > 0:
                dx_a, dy_a = current_pos_on_map_gps[0] - prev_pos_on_map_gps[0], current_pos_on_map_gps[1] - prev_pos_on_map_gps[1]
                magnitude = np.sqrt(dx_a*dx_a + dy_a*dy_a)
                if magnitude > 0:
                    udx, udy = dx_a / magnitude, dy_a / magnitude
                    arrow_len = 18
                    arrow_tip_x, arrow_tip_y = int(current_pos_on_map_gps[0] + udx * arrow_len), int(current_pos_on_map_gps[1] + udy * arrow_len)
                    if (0 <= arrow_tip_x < mini_mapa_w and 0 <= arrow_tip_y < mini_mapa_h):
                         cv2.arrowedLine(mini_mapa, current_pos_on_map_gps, (arrow_tip_x, arrow_tip_y), arrow_color_gps, 2, tipLength=0.4)
    
    if frame_corregido.shape[0] >= mini_mapa_h + 10 and frame_corregido.shape[1] >= mini_mapa_w + 10:
        frame_corregido[10:mini_mapa_h+10, frame_corregido.shape[1]-mini_mapa_w-10:frame_corregido.shape[1]-10] = mini_mapa
    
    # --- INICIO: NUEVO DISEÑO DE VISUALIZACIÓN DE INFORMACIÓN ---
    font = cv2.FONT_HERSHEY_SIMPLEX; line_type = cv2.LINE_AA
    font_scale_top = 0.8; font_scale_info_boxes = 0.7; font_scale_bottom_center = 0.8
    font_color_general = (255, 255, 255); font_color_yellow = (0, 255, 255); thickness_general = 2

    (text_w_video_label, text_h_video_label), _ = cv2.getTextSize("Video analizado:", font, font_scale_top, thickness_general)
    (text_w_video_value, _), _ = cv2.getTextSize(os.path.basename(ruta_video), font, font_scale_top, thickness_general)
    (text_w_code_label, text_h_code_label), _ = cv2.getTextSize("Codigo:", font, font_scale_top, thickness_general)
    (text_w_code_value, _), _ = cv2.getTextSize(script_name, font, font_scale_top, thickness_general)

    center_x_f = int(frame_width / 2); start_y_f = 35; line_spacing_f = 12

    y_video_f = start_y_f
    x_video_label_f = center_x_f - (text_w_video_label + text_w_video_value + 5) // 2
    x_video_value_f = x_video_label_f + text_w_video_label + 5
    cv2.putText(frame_corregido, "Video analizado:", (x_video_label_f, y_video_f), font, font_scale_top, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, "Video analizado:", (x_video_label_f, y_video_f), font, font_scale_top, font_color_yellow, thickness_general, line_type)
    cv2.putText(frame_corregido, os.path.basename(ruta_video), (x_video_value_f, y_video_f), font, font_scale_top, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, os.path.basename(ruta_video), (x_video_value_f, y_video_f), font, font_scale_top, font_color_general, thickness_general, line_type)

    y_pos_f = y_video_f + text_h_video_label + line_spacing_f
    fixed_text_vo = "Pos. VO:"
    value_text_vo = f"X:{t_f[0][0]:.1f} Z:{t_f[2][0]:.1f}" 
    fixed_text_gps = "| Pos. GPS:"
    x_real_gps, y_real_gps = (round(float(gps_posiciones_vehiculo_raw[-1][1]),5), round(float(gps_posiciones_vehiculo_raw[-1][0]),5)) if gps_posiciones_vehiculo_raw else (0.0,0.0)
    value_text_gps = f"Lon:{x_real_gps:.5f} Lat:{y_real_gps:.5f}"

    (text_w_fixed_vo, _), _ = cv2.getTextSize(fixed_text_vo, font, font_scale_top, thickness_general)
    (text_w_value_vo, _), _ = cv2.getTextSize(value_text_vo, font, font_scale_top, thickness_general)
    (text_w_fixed_gps, _), _ = cv2.getTextSize(fixed_text_gps, font, font_scale_top, thickness_general)
    (text_w_value_gps, _), _ = cv2.getTextSize(value_text_gps, font, font_scale_top, thickness_general)
    total_width_pos = text_w_fixed_vo + text_w_value_vo + text_w_fixed_gps + text_w_value_gps + 30
    x_fixed_vo = center_x_f - total_width_pos // 2
    x_value_vo = x_fixed_vo + text_w_fixed_vo + 5
    x_fixed_gps = x_value_vo + text_w_value_vo + 20
    x_value_gps = x_fixed_gps + text_w_fixed_gps + 5

    cv2.putText(frame_corregido, fixed_text_vo, (x_fixed_vo, y_pos_f), font, font_scale_top, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, fixed_text_vo, (x_fixed_vo, y_pos_f), font, font_scale_top, font_color_yellow, thickness_general, line_type)
    cv2.putText(frame_corregido, value_text_vo, (x_value_vo, y_pos_f), font, font_scale_top, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, value_text_vo, (x_value_vo, y_pos_f), font, font_scale_top, font_color_general, thickness_general, line_type)
    cv2.putText(frame_corregido, fixed_text_gps, (x_fixed_gps, y_pos_f), font, font_scale_top, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, fixed_text_gps, (x_fixed_gps, y_pos_f), font, font_scale_top, font_color_yellow, thickness_general, line_type)
    cv2.putText(frame_corregido, value_text_gps, (x_value_gps, y_pos_f), font, font_scale_top, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, value_text_gps, (x_value_gps, y_pos_f), font, font_scale_top, font_color_general, thickness_general, line_type)

    y_code_f = y_pos_f + text_h_code_label + line_spacing_f 
    x_code_label_f = center_x_f - (text_w_code_label + text_w_code_value + 5) // 2
    x_code_value_f = x_code_label_f + text_w_code_label + 5
    cv2.putText(frame_corregido, "Codigo:", (x_code_label_f, y_code_f), font, font_scale_top, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, "Codigo:", (x_code_label_f, y_code_f), font, font_scale_top, font_color_yellow, thickness_general, line_type)
    cv2.putText(frame_corregido, script_name, (x_code_value_f, y_code_f), font, font_scale_top, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, script_name, (x_code_value_f, y_code_f), font, font_scale_top, font_color_general, thickness_general, line_type)

    current_date_str = datetime.now().strftime("%d-%m-%Y"); date_label = "Fecha analisis:"
    (text_w_date_label, _), _ = cv2.getTextSize(date_label, font, font_scale_info_boxes, thickness_general)
    (text_w_date_value, _), _ = cv2.getTextSize(current_date_str, font, font_scale_info_boxes, thickness_general)
    x_date_label = zona_x_inicio; x_date_value = x_date_label + text_w_date_label + 5
    y_date = zona_y_inicio - 18
    cv2.putText(frame_corregido, date_label, (x_date_label, y_date), font, font_scale_info_boxes, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, date_label, (x_date_label, y_date), font, font_scale_info_boxes, font_color_yellow, thickness_general, line_type)
    cv2.putText(frame_corregido, current_date_str, (x_date_value, y_date), font, font_scale_info_boxes, (0,0,0), thickness_general + 2, line_type)
    cv2.putText(frame_corregido, current_date_str, (x_date_value, y_date), font, font_scale_info_boxes, font_color_general, thickness_general, line_type)

    box_abcd_x = 20; box_abcd_y_start = frame_height - 120; line_height_abcd = 26
    abcd_labels = ["RP:", "RM:", "MH:", "MM:"]
    abcd_values = [PROCESS_NOISE_MULTIPLIER, MEASUREMENT_NOISE_MULTIPLIER, MIN_HITS_TO_CONFIRM, MAX_MISSES_TO_DELETE]
    for i, (label, value) in enumerate(zip(abcd_labels, abcd_values)):
        y_abcd = box_abcd_y_start + i * line_height_abcd
        (text_w_label_abcd, _), _ = cv2.getTextSize(label, font, font_scale_info_boxes, thickness_general) 
        x_label_abcd = box_abcd_x; x_value_abcd = x_label_abcd + text_w_label_abcd + 5 
        cv2.putText(frame_corregido, label, (x_label_abcd, y_abcd), font, font_scale_info_boxes, (0,0,0), thickness_general + 2, line_type)
        cv2.putText(frame_corregido, label, (x_label_abcd, y_abcd), font, font_scale_info_boxes, font_color_yellow, thickness_general, line_type)
        cv2.putText(frame_corregido, str(value), (x_value_abcd, y_abcd), font, font_scale_info_boxes, (0,0,0), thickness_general + 2, line_type)
        cv2.putText(frame_corregido, str(value), (x_value_abcd, y_abcd), font, font_scale_info_boxes, font_color_general, thickness_general, line_type)

    frame_label = "Frame:"; time_label = "Tiempo:"
    (text_w_frame_label, _), _ = cv2.getTextSize(frame_label, font, font_scale_bottom_center, thickness_general)
    (text_w_frame_value, _), _ = cv2.getTextSize(str(frame_idx), font, font_scale_bottom_center, thickness_general)
    (text_w_time_label, _), _ = cv2.getTextSize(time_label, font, font_scale_bottom_center, thickness_general)
    (text_w_time_value, _), _ = cv2.getTextSize(f"{tiempo_actual:.1f}s", font, font_scale_bottom_center, thickness_general)
    
    max_width_abcd_labels = 0
    for lbl_idx, lbl_val_str in enumerate(abcd_labels):
        (w_lbl, _),_ = cv2.getTextSize(lbl_val_str, font, font_scale_info_boxes, thickness_general)
        (w_val, _),_ = cv2.getTextSize(str(abcd_values[lbl_idx]), font, font_scale_info_boxes, thickness_general)
        max_width_abcd_labels = max(max_width_abcd_labels, w_lbl + w_val + 5)

    frame_time_x = box_abcd_x + max_width_abcd_labels + 40
    frame_time_y = frame_height - 30
    x_frame_label = frame_time_x; x_frame_value = x_frame_label + text_w_frame_label + 5
    x_time_label = x_frame_value + text_w_frame_value + 20; x_time_value = x_time_label + text_w_time_label + 5
    cv2.putText(frame_corregido, frame_label, (x_frame_label, frame_time_y), font, font_scale_bottom_center, (0,0,0), thickness_general+2, line_type)
    cv2.putText(frame_corregido, frame_label, (x_frame_label, frame_time_y), font, font_scale_bottom_center, font_color_yellow, thickness_general, line_type)
    cv2.putText(frame_corregido, str(frame_idx), (x_frame_value, frame_time_y), font, font_scale_bottom_center, (0,0,0), thickness_general+2, line_type)
    cv2.putText(frame_corregido, str(frame_idx), (x_frame_value, frame_time_y), font, font_scale_bottom_center, font_color_general, thickness_general, line_type)
    cv2.putText(frame_corregido, time_label, (x_time_label, frame_time_y), font, font_scale_bottom_center, (0,0,0), thickness_general+2, line_type)
    cv2.putText(frame_corregido, time_label, (x_time_label, frame_time_y), font, font_scale_bottom_center, font_color_yellow, thickness_general, line_type)
    cv2.putText(frame_corregido, f"{tiempo_actual:.1f}s", (x_time_value, frame_time_y), font, font_scale_bottom_center, (0,0,0), thickness_general+2, line_type)
    cv2.putText(frame_corregido, f"{tiempo_actual:.1f}s", (x_time_value, frame_time_y), font, font_scale_bottom_center, font_color_general, thickness_general, line_type)

    box_e_margin_right = 20; box_e_item_height = 26; box_e_y_start = frame_height - 120
    font_color_e = font_color_general # Color for Tacha recent IDs
    font_color_f = (0, 165, 255) # Orange for Captafaro recent IDs

    # Display recent Tacha IDs
    for i, confirmed_info in enumerate(reversed(ultimas_5_ids_confirmadas)):
        a_gps, b_gps = (round(float(gps_posiciones_vehiculo_raw[-1][1]),2), round(float(gps_posiciones_vehiculo_raw[-1][0]),2)) if gps_posiciones_vehiculo_raw else (0.0, 0.0)
        text_to_display_e = f"T_ID: {confirmed_info['id']},PS x:{confirmed_info['cx']}, y:{confirmed_info['cy']}; PR Lon:{a_gps:.2f}, Lat:{b_gps:.2f}"
        current_y_e = box_e_y_start + i * box_e_item_height
        (text_w_e, _), _ = cv2.getTextSize(text_to_display_e, font, font_scale_info_boxes, thickness_general)
        actual_text_x_e = frame_width - text_w_e - box_e_margin_right
        if current_y_e < frame_height - 5 and current_y_e > box_e_item_height :
             cv2.putText(frame_corregido, text_to_display_e, (actual_text_x_e, current_y_e), font, font_scale_info_boxes, (0,0,0), thickness_general+2, line_type)
             cv2.putText(frame_corregido, text_to_display_e, (actual_text_x_e, current_y_e), font, font_scale_info_boxes, font_color_e, thickness_general, line_type)

    # Display recent Captafaro IDs (adjust y_start if needed to avoid overlap or position below tachas)
    box_f_y_start = frame_height - 120 - (MAX_DISPLAY_IDS * box_e_item_height) - 10 # Position above tachas or adjust
    if len(ultimas_5_ids_confirmadas) == 0 : # if no tachas, captafaros start at box_e_y_start
        box_f_y_start = box_e_y_start

    for i, confirmed_info_c in enumerate(reversed(ultimas_5_ids_confirmadas_captafaros)):
        a_gps_c, b_gps_c = (round(float(gps_posiciones_vehiculo_raw[-1][1]),2), round(float(gps_posiciones_vehiculo_raw[-1][0]),2)) if gps_posiciones_vehiculo_raw else (0.0, 0.0)
        text_to_display_f = f"C_ID: {confirmed_info_c['id']},PS x:{confirmed_info_c['cx']}, y:{confirmed_info_c['cy']}; PR Lon:{a_gps_c:.2f}, Lat:{b_gps_c:.2f}"
        current_y_f = box_f_y_start + i * box_e_item_height 
        (text_w_f, _), _ = cv2.getTextSize(text_to_display_f, font, font_scale_info_boxes, thickness_general)
        actual_text_x_f = frame_width - text_w_f - box_e_margin_right
        if current_y_f < frame_height - 5 and current_y_f > box_e_item_height and current_y_f < box_e_y_start -5 : # Ensure it does not overlap with frame boundary or tacha IDs
             cv2.putText(frame_corregido, text_to_display_f, (actual_text_x_f, current_y_f), font, font_scale_info_boxes, (0,0,0), thickness_general+2, line_type)
             cv2.putText(frame_corregido, text_to_display_f, (actual_text_x_f, current_y_f), font, font_scale_info_boxes, font_color_f, thickness_general, line_type)
    # --- FIN: NUEVO DISEÑO DE VISUALIZACIÓN ---

    out.write(frame_corregido)
    cv2.imshow('Procesamiento en tiempo real', cv2.resize(frame_corregido, (0, 0), fx=0.5, fy=0.5))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

    if frame_idx % 100 == 0: print(f"Frame {frame_idx} procesado...")
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# --- ANÁLISIS DE PATRÓN DE COLOCACIÓN DE TACHAS ---
print("\n--- Iniciando Análisis de Patrón de Colocación de Tachas ---")
historial_solo_tachas = historial_confirmadas
total_detecciones_unicas_tachas_analisis = 0
summary_metrics_for_excel = [] 

if historial_solo_tachas:
    historial_tachas_ordenado = sorted(historial_solo_tachas, key=lambda t: (t['centro_transformado'][1], t['centro_transformado'][0]))
    total_detecciones_unicas_tachas_analisis = len(historial_tachas_ordenado)
    print(f"Total de detecciones de tachas únicas procesadas para análisis: {total_detecciones_unicas_tachas_analisis}")
    summary_metrics_for_excel.append({"Métrica": "Total de tachas únicas detectadas", "Valor": total_detecciones_unicas_tachas_analisis})
else:
    historial_tachas_ordenado = []
    print("No se encontraron tachas en el historial para analizar.")
    summary_metrics_for_excel.append({"Métrica": "Total de tachas únicas detectadas", "Valor": 0})
    summary_metrics_for_excel.append({"Métrica": "Advertencia", "Valor": "No se encontraron tachas para análisis."})

distancias_entre_tachas_detectadas_list = [] 
distancias_info_for_excel = [] 

if len(historial_tachas_ordenado) > 1:
    for i in range(1, len(historial_tachas_ordenado)):
        tacha_anterior, tacha_actual = historial_tachas_ordenado[i-1], historial_tachas_ordenado[i]
        distancia = np.linalg.norm(tacha_actual['centro_transformado'] - tacha_anterior['centro_transformado'])
        distancias_entre_tachas_detectadas_list.append(distancia)
        distancias_info_for_excel.append({
            'ID Tacha Anterior': tacha_anterior.get('tacha_id_secuencial', tacha_anterior.get('id')),
            'ID Tacha Actual': tacha_actual.get('tacha_id_secuencial', tacha_actual.get('id')),
            'Coord. Anterior (X_bev,Y_bev)': f"{tacha_anterior['centro_transformado'][0]:.2f}, {tacha_anterior['centro_transformado'][1]:.2f}",
            'Coord. Actual (X_bev,Y_bev)': f"{tacha_actual['centro_transformado'][0]:.2f}, {tacha_actual['centro_transformado'][1]:.2f}",
            'Distancia BEV (pix)': f"{distancia:.2f}"})
else:
    print("No hay suficientes tachas detectadas (se necesitan al menos 2) para calcular distancias.")
    summary_metrics_for_excel.append({"Métrica": "Cálculo de distancias", "Valor": "Insuficientes tachas (requiere >= 2)."})

media_espaciado_pix_bev, std_dev_espaciado_pix_bev = 0, 0
if distancias_entre_tachas_detectadas_list:
    media_espaciado_pix_bev = np.mean(distancias_entre_tachas_detectadas_list)
    std_dev_espaciado_pix_bev = np.std(distancias_entre_tachas_detectadas_list)
    min_espaciado_pix_bev = np.min(distancias_entre_tachas_detectadas_list)
    max_espaciado_pix_bev = np.max(distancias_entre_tachas_detectadas_list)
    print(f"\nMedia del espaciado (detectadas): {media_espaciado_pix_bev:.2f} píxeles (BEV)")
    print(f"Desviación estándar del espaciado (detectadas): {std_dev_espaciado_pix_bev:.2f} píxeles (BEV)")
    summary_metrics_for_excel.extend([
        {"Métrica": "Media espaciado (píxeles BEV)", "Valor": f"{media_espaciado_pix_bev:.2f}"},
        {"Métrica": "Std Dev espaciado (píxeles BEV)", "Valor": f"{std_dev_espaciado_pix_bev:.2f}"},
        {"Métrica": "Min espaciado (píxeles BEV)", "Valor": f"{min_espaciado_pix_bev:.2f}"},
        {"Métrica": "Max espaciado (píxeles BEV)", "Valor": f"{max_espaciado_pix_bev:.2f}"}])

tachas_faltantes_estimadas = []
if media_espaciado_pix_bev > 0 and historial_tachas_ordenado:
    print(f"\n--- Estimando Tachas Faltantes (usando media espaciado: {media_espaciado_pix_bev:.2f}) ---")
    tachas_faltantes_estimadas = detectar_tachas_faltantes(historial_tachas_ordenado, media_espaciado_pix_bev)
    print(f"Se estimaron {len(tachas_faltantes_estimadas)} tachas faltantes.")
    summary_metrics_for_excel.append({"Métrica": "Número de tachas faltantes estimadas", "Valor": len(tachas_faltantes_estimadas)})
else:
    print("No se pudo estimar tachas faltantes (media de espaciado no válida o no hay suficientes tachas detectadas).")
    summary_metrics_for_excel.append({"Métrica": "Estimación tachas faltantes", "Valor": "No realizada o 0."})

df_sheet1_summary = pd.DataFrame(summary_metrics_for_excel)
df_sheet1_distancias_detalle = pd.DataFrame(distancias_info_for_excel)

inventario_data_for_excel = []
for tacha_detectada in historial_tachas_ordenado:
    inventario_data_for_excel.append({
        'ID_Tacha': tacha_detectada.get('tacha_id_secuencial', tacha_detectada.get('id')),
        'Centro_Transformado_X': f"{tacha_detectada['centro_transformado'][0]:.2f}",
        'Centro_Transformado_Y': f"{tacha_detectada['centro_transformado'][1]:.2f}",
        'Frame_Referencia': tacha_detectada['frame'], 'Estado': 'Detectada'})
for tacha_faltante in tachas_faltantes_estimadas: 
    inventario_data_for_excel.append({
        'ID_Tacha': tacha_faltante['tacha_id_secuencial'],
        'Centro_Transformado_X': f"{tacha_faltante['centro_transformado'][0]:.2f}",
        'Centro_Transformado_Y': f"{tacha_faltante['centro_transformado'][1]:.2f}",
        'Frame_Referencia': tacha_faltante['frame'], 'Estado': tacha_faltante['Estado']})

df_sheet2_inventario_tachas = pd.DataFrame(inventario_data_for_excel)
if not df_sheet2_inventario_tachas.empty:
    df_sheet2_inventario_tachas['Centro_Transformado_Y_float'] = pd.to_numeric(df_sheet2_inventario_tachas['Centro_Transformado_Y'], errors='coerce')
    df_sheet2_inventario_tachas['Centro_Transformado_X_float'] = pd.to_numeric(df_sheet2_inventario_tachas['Centro_Transformado_X'], errors='coerce')
    df_sheet2_inventario_tachas = df_sheet2_inventario_tachas.sort_values(by=['Centro_Transformado_Y_float', 'Centro_Transformado_X_float']).drop(columns=['Centro_Transformado_Y_float', 'Centro_Transformado_X_float'])

log_confirmadas_for_excel = []
historial_confirmadas_sorted_log = sorted(historial_confirmadas, key=lambda x: x['tacha_id_secuencial'])
for i, tracker_info in enumerate(historial_confirmadas_sorted_log):
    current_data = {'Internal_Tracker_ID': tracker_info['internal_id'],
        'Tacha_ID_Secuencial': tracker_info['tacha_id_secuencial'],
        'Centro_Transformado_X': f"{tracker_info['centro_transformado'][0]:.2f}",
        'Centro_Transformado_Y': f"{tracker_info['centro_transformado'][1]:.2f}",
        'Frame_Confirmacion': tracker_info['frame'], 'Clase_Detectada': tracker_info['clase'],
        'Estado_Tracker': 'Confirmado', 'ID_Secuencial_Anterior': 'N/A', 
        'Coord_Anterior_X': 'N/A', 'Coord_Anterior_Y': 'N/A', 'Distancia_BEV_pix': 'N/A'}
    if i > 0:
        prev_tracker_info = historial_confirmadas_sorted_log[i-1]
        current_coords, prev_coords = tracker_info['centro_transformado'], prev_tracker_info['centro_transformado']
        distancia = np.linalg.norm(current_coords - prev_coords)
        current_data.update({'ID_Secuencial_Anterior': prev_tracker_info['tacha_id_secuencial'],
                             'Coord_Anterior_X': f"{prev_coords[0]:.2f}", 'Coord_Anterior_Y': f"{prev_coords[1]:.2f}",
                             'Distancia_BEV_pix': f"{distancia:.2f}"})
    log_confirmadas_for_excel.append(current_data)
df_sheet3_log_confirmadas = pd.DataFrame(log_confirmadas_for_excel)
# --- FIN ANÁLISIS TACHAS ---


# --- ANÁLISIS DE PATRÓN DE COLOCACIÓN DE CAPTAFAROS --- [cite: 87]
print("\n--- Iniciando Análisis de Patrón de Colocación de Captafaros ---") # [cite: 88]
historial_solo_captafaros = historial_confirmadas_captafaros # [cite: 88]
total_detecciones_unicas_captafaros_analisis = 0 # [cite: 88]
summary_metrics_for_excel_captafaros = [] # [cite: 88]

if historial_solo_captafaros: # [cite: 88]
    historial_captafaros_ordenado = sorted(historial_solo_captafaros, key=lambda t: (t['centro_transformado'][1], t['centro_transformado'][0])) # [cite: 88]
    total_detecciones_unicas_captafaros_analisis = len(historial_captafaros_ordenado) # [cite: 88]
    print(f"Total de detecciones de captafaros únicas procesadas para análisis: {total_detecciones_unicas_captafaros_analisis}") # [cite: 88]
    summary_metrics_for_excel_captafaros.append({"Métrica": "Total de captafaros únicas detectadas", "Valor": total_detecciones_unicas_captafaros_analisis}) # [cite: 88]
else: # [cite: 88]
    historial_captafaros_ordenado = [] # [cite: 88]
    print("No se encontraron captafaros en el historial para analizar.") # [cite: 88]
    summary_metrics_for_excel_captafaros.append({"Métrica": "Total de captafaros únicas detectadas", "Valor": 0}) # [cite: 88]
    summary_metrics_for_excel_captafaros.append({"Métrica": "Advertencia", "Valor": "No se encontraron captafaros para análisis."}) # [cite: 89]

distancias_entre_captafaros_detectadas_list = [] # [cite: 89]
distancias_info_for_excel_captafaros = [] # [cite: 89]

if len(historial_captafaros_ordenado) > 1: # [cite: 89]
    for i in range(1, len(historial_captafaros_ordenado)): # [cite: 89]
        captafaro_anterior, captafaro_actual = historial_captafaros_ordenado[i-1], historial_captafaros_ordenado[i] # [cite: 89]
        distancia = np.linalg.norm(captafaro_actual['centro_transformado'] - captafaro_anterior['centro_transformado']) # [cite: 89]
        distancias_entre_captafaros_detectadas_list.append(distancia) # [cite: 89]
        distancias_info_for_excel_captafaros.append({ # [cite: 89]
            'ID Captafaro Anterior': captafaro_anterior.get('captafaro_id_secuencial', captafaro_anterior.get('id')), # [cite: 89]
            'ID Captafaro Actual': captafaro_actual.get('captafaro_id_secuencial', captafaro_actual.get('id')), # [cite: 89]
            'Coord. Anterior (X_bev,Y_bev)': f"{captafaro_anterior['centro_transformado'][0]:.2f}, {captafaro_anterior['centro_transformado'][1]:.2f}", # [cite: 91]
            'Coord. Actual (X_bev,Y_bev)': f"{captafaro_actual['centro_transformado'][0]:.2f}, {captafaro_actual['centro_transformado'][1]:.2f}", # [cite: 92]
            'Distancia BEV (pix)': f"{distancia:.2f}"}) # [cite: 92]
else: # [cite: 92]
    print("No hay suficientes captafaros detectadas (se necesitan al menos 2) para calcular distancias.") # [cite: 92]
    summary_metrics_for_excel_captafaros.append({"Métrica": "Cálculo de distancias Captafaros", "Valor": "Insuficientes captafaros (requiere >= 2)."}) # [cite: 92]

media_espaciado_pix_bev_captafaros, std_dev_espaciado_pix_bev_captafaros = 0, 0 # [cite: 92]
if distancias_entre_captafaros_detectadas_list: # [cite: 92]
    media_espaciado_pix_bev_captafaros = np.mean(distancias_entre_captafaros_detectadas_list) # [cite: 92]
    std_dev_espaciado_pix_bev_captafaros = np.std(distancias_entre_captafaros_detectadas_list) # [cite: 92]
    min_espaciado_pix_bev_captafaros = np.min(distancias_entre_captafaros_detectadas_list) # [cite: 92]
    max_espaciado_pix_bev_captafaros = np.max(distancias_entre_captafaros_detectadas_list) # [cite: 92]
    print(f"\nMedia del espaciado de captafaros (detectadas): {media_espaciado_pix_bev_captafaros:.2f} píxeles (BEV)") # [cite: 92]
    print(f"Desviación estándar del espaciado de captafaros (detectadas): {std_dev_espaciado_pix_bev_captafaros:.2f} píxeles (BEV)") # [cite: 92]
    summary_metrics_for_excel_captafaros.extend([ # [cite: 93]
        {"Métrica": "Media espaciado Captafaros (píxeles BEV)", "Valor": f"{media_espaciado_pix_bev_captafaros:.2f}"}, # [cite: 93]
        {"Métrica": "Std Dev espaciado Captafaros (píxeles BEV)", "Valor": f"{std_dev_espaciado_pix_bev_captafaros:.2f}"}, # [cite: 93]
        {"Métrica": "Min espaciado Captafaros (píxeles BEV)", "Valor": f"{min_espaciado_pix_bev_captafaros:.2f}"}, # [cite: 93]
        {"Métrica": "Max espaciado Captafaros (píxeles BEV)", "Valor": f"{max_espaciado_pix_bev_captafaros:.2f}"}]) # [cite: 93]

captafaros_faltantes_estimadas = [] # [cite: 93]
if media_espaciado_pix_bev_captafaros > 0 and historial_captafaros_ordenado: # [cite: 93]
    print(f"\n--- Estimando Captafaros Faltantes (usando media espaciado: {media_espaciado_pix_bev_captafaros:.2f}) ---") # [cite: 93]
    captafaros_faltantes_estimadas = detectar_captafaros_faltantes(historial_captafaros_ordenado, media_espaciado_pix_bev_captafaros) # [cite: 33, 93]
    print(f"Se estimaron {len(captafaros_faltantes_estimadas)} captafaros faltantes.") # [cite: 93]
    summary_metrics_for_excel_captafaros.append({"Métrica": "Número de captafaros faltantes estimadas", "Valor": len(captafaros_faltantes_estimadas)}) # [cite: 94]
else: # [cite: 94]
    print("No se pudo estimar captafaros faltantes (media de espaciado no válida o no hay suficientes captafaros detectadas).") # [cite: 94]
    summary_metrics_for_excel_captafaros.append({"Métrica": "Estimación captafaros faltantes", "Valor": "No realizada o 0."}) # [cite: 94]

df_captafaros_sheet1_summary = pd.DataFrame(summary_metrics_for_excel_captafaros) # [cite: 94]
df_captafaros_sheet1_distancias_detalle = pd.DataFrame(distancias_info_for_excel_captafaros) # [cite: 94]

inventario_data_for_excel_captafaros = [] # [cite: 94]
for captafaro_detectado in historial_captafaros_ordenado: # [cite: 94]
    inventario_data_for_excel_captafaros.append({ # [cite: 94]
        'ID_Captafaro': captafaro_detectado.get('captafaro_id_secuencial', captafaro_detectado.get('id')), # [cite: 94]
        'Centro_Transformado_X': f"{captafaro_detectado['centro_transformado'][0]:.2f}", # [cite: 94]
        'Centro_Transformado_Y': f"{captafaro_detectado['centro_transformado'][1]:.2f}", # [cite: 94]
        'Frame_Referencia': captafaro_detectado['frame'], 'Estado': 'Detectada'}) # [cite: 94]
for captafaro_faltante in captafaros_faltantes_estimadas: # [cite: 94]
    inventario_data_for_excel_captafaros.append({ # [cite: 95]
        'ID_Captafaro': captafaro_faltante['captafaro_id_secuencial'], # [cite: 95]
        'Centro_Transformado_X': f"{captafaro_faltante['centro_transformado'][0]:.2f}", # [cite: 95]
        'Centro_Transformado_Y': f"{captafaro_faltante['centro_transformado'][1]:.2f}", # [cite: 95]
        'Frame_Referencia': captafaro_faltante['frame'], 'Estado': captafaro_faltante['Estado']}) # [cite: 95]

df_captafaros_sheet2_inventario = pd.DataFrame(inventario_data_for_excel_captafaros) # [cite: 95]
if not df_captafaros_sheet2_inventario.empty: # [cite: 95]
    df_captafaros_sheet2_inventario['Centro_Transformado_Y_float'] = pd.to_numeric(df_captafaros_sheet2_inventario['Centro_Transformado_Y'], errors='coerce') # [cite: 95]
    df_captafaros_sheet2_inventario['Centro_Transformado_X_float'] = pd.to_numeric(df_captafaros_sheet2_inventario['Centro_Transformado_X'], errors='coerce') # [cite: 95]
    df_captafaros_sheet2_inventario = df_captafaros_sheet2_inventario.sort_values(by=['Centro_Transformado_Y_float', 'Centro_Transformado_X_float']).drop(columns=['Centro_Transformado_Y_float', 'Centro_Transformado_X_float']) # [cite: 95]

log_confirmadas_for_excel_captafaros = [] # [cite: 95]
# Ensure captafaro_id_secuencial exists, otherwise sort by internal_id as fallback
historial_confirmadas_captafaros_sorted_log = sorted(
    historial_confirmadas_captafaros, 
    key=lambda x: x.get('captafaro_id_secuencial', x['internal_id']) 
)
for i, tracker_info in enumerate(historial_confirmadas_captafaros_sorted_log): # renamed variable
    current_data = {'Internal_Tracker_ID': tracker_info['internal_id'], # [cite: 95]
        'Captafaro_ID_Secuencial': tracker_info.get('captafaro_id_secuencial', 'N/A'), # Use .get for safety
        'Centro_Transformado_X': f"{tracker_info['centro_transformado'][0]:.2f}", # [cite: 96]
        'Centro_Transformado_Y': f"{tracker_info['centro_transformado'][1]:.2f}", # [cite: 96]
        'Frame_Confirmacion': tracker_info['frame'], 'Clase_Detectada': tracker_info['clase'], # [cite: 96]
        'Estado_Tracker': 'Confirmado', 'ID_Secuencial_Anterior': 'N/A', # [cite: 96]
        'Coord_Anterior_X': 'N/A', 'Coord_Anterior_Y': 'N/A', 'Distancia_BEV_pix': 'N/A'} # [cite: 96]
    if i > 0: # [cite: 96]
        prev_tracker_info = historial_confirmadas_captafaros_sorted_log[i-1] # renamed variable
        current_coords, prev_coords = tracker_info['centro_transformado'], prev_tracker_info['centro_transformado'] # [cite: 96]
        distancia = np.linalg.norm(current_coords - prev_coords) # [cite: 96]
        current_data.update({'ID_Secuencial_Anterior': prev_tracker_info.get('captafaro_id_secuencial', 'N/A'), # Use .get for safety [cite: 96]
                             'Coord_Anterior_X': f"{prev_coords[0]:.2f}", 'Coord_Anterior_Y': f"{prev_coords[1]:.2f}", # [cite: 97]
                             'Distancia_BEV_pix': f"{distancia:.2f}"}) # [cite: 97]
    log_confirmadas_for_excel_captafaros.append(current_data) # [cite: 97]
df_captafaros_sheet3_log_confirmadas = pd.DataFrame(log_confirmadas_for_excel_captafaros) # [cite: 97]
# --- FIN ANÁLISIS CAPTAFAROS ---


fecha_actual_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
nombre_archivo_excel = f"{fecha_actual_str}_AnalisisTachasCaptafarosFP_SSGL.xlsx" # Nombre modificado
try: # [cite: 97]
    with pd.ExcelWriter(nombre_archivo_excel, engine='openpyxl') as writer: # [cite: 97]
        df_sheet1_summary.to_excel(writer, sheet_name='Resumen Tachas y Distancias', index=False, header=True) # [cite: 97]
        if not df_sheet1_distancias_detalle.empty: # [cite: 98]
            writer.sheets['Resumen Tachas y Distancias'].cell(row=len(df_sheet1_summary) + 3, column=1, value="Detalle de Distancias entre Tachas Detectadas (Ordenadas por BEV)") # [cite: 98]
            df_sheet1_distancias_detalle.to_excel(writer, sheet_name='Resumen Tachas y Distancias', index=False, startrow=len(df_sheet1_summary) + 4) # [cite: 98]
        df_sheet2_inventario_tachas.to_excel(writer, sheet_name='Inventario Tachas (Det_Falt)', index=False) # [cite: 98]
        df_sheet3_log_confirmadas.to_excel(writer, sheet_name='Log Tachas Confirmadas', index=False) # [cite: 98]

        # Add Captafaro sheets [cite: 98]
        df_captafaros_sheet1_summary.to_excel(writer, sheet_name='Resumen Captafaros y Dist.', index=False, header=True) # [cite: 98]
        if not df_captafaros_sheet1_distancias_detalle.empty: # [cite: 99]
            writer.sheets['Resumen Captafaros y Dist.'].cell(row=len(df_captafaros_sheet1_summary) + 3, column=1, value="Detalle de Distancias entre Captafaros Detectados (Ordenadas por BEV)") # [cite: 99]
            df_captafaros_sheet1_distancias_detalle.to_excel(writer, sheet_name='Resumen Captafaros y Dist.', index=False, startrow=len(df_captafaros_sheet1_summary) + 4) # [cite: 99]
        df_captafaros_sheet2_inventario.to_excel(writer, sheet_name='Inventario Captafaros (Det_Falt)', index=False) # [cite: 99]
        df_captafaros_sheet3_log_confirmadas.to_excel(writer, sheet_name='Log Captafaros Confirmados', index=False) # [cite: 99]

    print(f"\nInforme de análisis completo guardado en: '{nombre_archivo_excel}'") # [cite: 99]
except Exception as e: # [cite: 99]
    print(f"\nError al guardar el informe de análisis en Excel: {e}") # [cite: 100]


# Gráfico Comparativo de Trayectorias (VO vs GPS)
if posiciones_vehiculo and gps_posiciones_vehiculo_raw: 
    odo_np = np.array(posiciones_vehiculo)
    gps_np_raw = np.array(gps_posiciones_vehiculo_raw) 

    plt.figure(figsize=(12, 10))
    
    plt.plot(odo_np[:, 0], odo_np[:, 1], 'g-', label='Odometría Visual (X, Z relativos)')

    if gps_np_raw.ndim == 2 and gps_np_raw.shape[1] == 2:
        gps_lon_origin = gps_np_raw[0,1]
        gps_lat_origin = gps_np_raw[0,0]
        plt.plot((gps_np_raw[:, 1] - gps_lon_origin) * 1e5 , (gps_np_raw[:, 0] - gps_lat_origin) * 1e5, 'b--', label='GPS (delta Lon*1e5, delta Lat*1e5)')
    else:
        print("Datos GPS (gps_posiciones_vehiculo_raw) no tienen el formato esperado para graficar.")
        
    plt.title('Comparación de Trayectorias (VO y GPS relativo)')
    plt.xlabel('Eje X / Delta Longitud (escalado)')
    plt.ylabel('Eje Z / Delta Latitud (escalado)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') 
    try:
        plt.savefig('trayectoria_comparativa_vo_gps_mejorada.png', dpi=300)
        print("Gráfico de trayectoria comparativa guardado.")
    except Exception as e:
        print(f"Error al guardar el gráfico de trayectoria: {e}")
else:
    print("No hay suficientes datos de odometría o GPS para graficar la trayectoria comparativa.")

print("\nProcesamiento finalizado.")