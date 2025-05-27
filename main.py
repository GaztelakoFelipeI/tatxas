# Mejora de odometría visual con acumulación de rotación y translación coherente
# Código realizado por Felipe Pereira Alarcón
# MODIFICADO: Sistema de seguimiento con Filtro de Kalman y Máquina de Estados

import cv2
import numpy as np
import torch
# import statistics # No se usa explícitamente, se puede quitar si no es para debug futuro
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from scipy.optimize import linear_sum_assignment

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
        
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],   # x_new = x_old + vx
                                              [0, 1, 0, 1],   # y_new = y_old + vy
                                              [0, 0, 1, 0],   # vx_new = vx_old
                                              [0, 0, 0, 1]], np.float32)
        
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],  # Medimos solo la posición x
                                               [0, 1, 0, 0]], np.float32) # Medimos solo la posición y

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.statePost = np.array([centro_img[0], centro_img[1], 0, 0], dtype=np.float32)

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

MIN_HITS_TO_CONFIRM = 3
MAX_MISSES_TO_DELETE = 5

### NUEVO ### Parámetros para detección de tachas faltantes en tiempo real
REALTIME_EXPECTED_SPACING_BEV = 60.0  # Distancia esperada (en píxeles BEV) entre tachas. ¡AJUSTAR ESTE VALOR!
REALTIME_MIN_SPACING_BEV = 20.0      # Distancia mínima para considerar que no son la misma tacha o un error
REALTIME_SPACING_TOLERANCE_FACTOR = 1.75 # Si la distancia es > ESPERADA * FACTOR, se infiere faltante
MIN_STUDS_FOR_REALTIME_PATTERN = 3   # Mínimo de tachas activas confirmadas para intentar detectar faltantes
# MAX_ALERT_DURATION_FRAMES se definirá después de obtener 'fps'

# Corrección de distorsión
K = np.array([[1500, 0, 1352], [0, 1500, 760], [0, 0, 1]])
D = np.array([-0.25, 0.03, 0, 0])

# Odometría visual con acumulación
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
prev_gray, prev_kp, prev_des = None, None, None
R_f = np.eye(3)
t_f = np.zeros((3, 1))
posiciones_vehiculo = [np.array([0.0, 0.0])]

# Trayectoria GPS
try:
    gps_data = pd.read_excel("metadata/3.2 - 01_04 Tramo B1-B2.xlsx")
    gps_trayectoria = gps_data[['Latitud', 'Longitud']].values
    gps_tiempos = gps_data['Tiempo'].values
    gps_posiciones_vehiculo = []
except FileNotFoundError:
    print("Archivo GPS 'metadata/3.2 - 01_04 Tramo B1-B2.xlsx' no encontrado.")
    gps_data = None


# Funciones auxiliares
def transformar_punto(punto, matriz):
    punto_homog = np.array([[punto[0], punto[1], 1.0]]).T
    transformado = np.dot(matriz, punto_homog)
    if transformado[2] == 0 or np.isclose(transformado[2], 0): # Evitar división por cero
        return np.array([float('inf'), float('inf')])
    transformado /= transformado[2]
    return transformado[0:2].flatten()

### NUEVA FUNCIÓN ###
def transformar_punto_inv(punto_bev, matriz_inversa):
    """Transforma un punto de la vista de pájaro (BEV) de vuelta a coordenadas de imagen."""
    punto_homog_bev = np.array([[punto_bev[0], punto_bev[1], 1.0]]).T
    transformado_img_homog = np.dot(matriz_inversa, punto_homog_bev)
    if transformado_img_homog[2] == 0 or np.isclose(transformado_img_homog[2], 0):
        return np.array([-1.0, -1.0]) # Indicar punto inválido o fuera de proyección razonable
    transformado_img_homog /= transformado_img_homog[2]
    return transformado_img_homog[0:2].flatten()

def detectar_objetos(frame):
    resultados_c = model_captafaros(frame, device=device)
    resultados_t = model_tachas(frame, device=device)
    resultados_s = model_senaleticas(frame, device=device)
    return (resultados_c[0].boxes.data.cpu().numpy(),
            resultados_t[0].boxes.data.cpu().numpy(),
            resultados_s[0].boxes.data.cpu().numpy())

def filtrar_detecciones(objetos, otros1, otros2, x0, x1, y0, y1, current_frame_idx):
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
        if height_b == 0: continue # Evitar división por cero
        aspect_ratio_b = width_b / height_b
        if not (MIN_TACHA_ASPECT_RATIO <= aspect_ratio_b <= MAX_TACHA_ASPECT_RATIO): continue
        
        demasiado_cerca_otros = False
        for x1o, y1o, x2o, y2o, *_ in otros1:
            cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2
            if np.linalg.norm([cx - cxo, cy - cyo]) < 30:
                demasiado_cerca_otros = True
                break
        if demasiado_cerca_otros: continue
        for x1o, y1o, x2o, y2o, *_ in otros2:
            cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2
            if np.linalg.norm([cx - cxo, cy - cyo]) < 30:
                demasiado_cerca_otros = True
                break
        if demasiado_cerca_otros: continue

        filtrados_detailed.append({
            'cx': cx, 'cy': cy, 'frame_idx': current_frame_idx
        })
    return filtrados_detailed

def detectar_tachas_faltantes(tachas_ordenadas, media_espaciado_bev, umbral_multiplicador=1.75):
    faltantes = []
    if not tachas_ordenadas or media_espaciado_bev <= 0:
        return faltantes
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
                        'frame': tacha_actual['frame'],
                        'clase': 'tacha_faltante_estimada',
                        'Estado': 'Faltante'
                    })
    return faltantes

# Video
ruta_video = 'videos/3.2 - 01_04 Tramo B1-B2.MP4'
output_video = '3.2 - 01_04 Tramo B1-B2_resultadoTransformado_conAnalisisTachas_NN_RealTimeAlert.MP4' # Modificado nombre salida
cap = cv2.VideoCapture(ruta_video)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el video {ruta_video}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

### NUEVO ### Definición de MAX_ALERT_DURATION_FRAMES después de obtener fps
MAX_ALERT_DURATION_FRAMES = int(fps * 2) if fps > 0 else 60 # Duración de la alerta en frames (ej. 2 segundos)


# --- Definición de Zona de Interés (ROI) y Transformación de Perspectiva ---
zona_x_inicio = int(frame_width * 0.20)
zona_x_fin = int(frame_width * 0.50)
zona_y_inicio = int(frame_height * 0.55)
zona_y_fin = frame_height
pts_origen = np.float32([[zona_x_inicio, zona_y_inicio], [zona_x_fin, zona_y_inicio], [zona_x_inicio, zona_y_fin], [zona_x_fin, zona_y_fin]])
ancho_transformado, alto_transformado = 400, 600 # Dimensiones del BEV
pts_destino = np.float32([[0, 0], [ancho_transformado, 0], [0, alto_transformado], [ancho_transformado, alto_transformado]])
M = cv2.getPerspectiveTransform(pts_origen, pts_destino)
M_inv = cv2.getPerspectiveTransform(pts_destino, pts_origen)

# --- Inicialización de variables para el bucle ---
frame_idx = 0
active_trackers = []
next_tacha_id = 0
next_sequential_confirmed_id = 1 
historial_confirmadas = []

ultimas_5_ids_confirmadas = []
MAX_DISPLAY_IDS = 5

### NUEVO ### Para alertas de tachas faltantes en tiempo real
active_missing_stud_notifications = [] # Lista de alertas activas

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    tiempo_actual = frame_idx / fps if fps > 0 else 0
    frame_corregido = cv2.undistort(frame, K, D)

    # --- INICIO DEL NUEVO FLUJO DE TRACKING ---
    # 1. DETECCIÓN
    captafaros_raw, tachas_raw, senaleticas_raw = detectar_objetos(frame_corregido)
    detecciones_actuales = filtrar_detecciones(
        tachas_raw, captafaros_raw, senaleticas_raw,
        zona_x_inicio, zona_x_fin, zona_y_inicio, zona_y_fin,
        frame_idx
    )

    # 2. PREDICCIÓN
    for tracker in active_trackers:
        tracker.predict()

    # 3. ASOCIACIÓN
    # ... (tu código de asociación existente) ...
    unmatched_detections = detecciones_actuales
    if active_trackers and detecciones_actuales:
        cost_matrix = np.zeros((len(active_trackers), len(detecciones_actuales)))
        for t, tracker_obj in enumerate(active_trackers): # Renombrado para evitar conflicto con variable global
            for d, det in enumerate(detecciones_actuales):
                pred_x, pred_y = tracker_obj.kf.statePost[0], tracker_obj.kf.statePost[1]
                det_x, det_y = det['cx'], det['cy']
                cost_matrix[t, d] = np.linalg.norm([pred_x - det_x, pred_y - det_y])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        max_dist_threshold = 50 
        matched_indices = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_dist_threshold:
                tracker_obj = active_trackers[r] # Usar tracker_obj
                det = detecciones_actuales[c]
                
                centro_img_det = np.array([det['cx'], det['cy']])
                centro_transformado_det = transformar_punto(centro_img_det, M)
                tracker_obj.update(centro_img_det, centro_transformado_det) # Usar tracker_obj
                matched_indices.append(c)
        
        unmatched_detections = [d for i, d in enumerate(detecciones_actuales) if i not in matched_indices]


    # 4b. CREACIÓN DE NUEVOS TRACKERS
    for det in unmatched_detections:
        centro_img_new = np.array([det['cx'], det['cy']])
        centro_transformado_new = transformar_punto(centro_img_new, M)
        new_tracker = TachaTracker(next_tacha_id, centro_img_new, centro_transformado_new, frame_idx)    
        active_trackers.append(new_tracker)
        next_tacha_id += 1

    # 5. MANEJO DE ESTADOS Y LIMPIEZA
    final_trackers = []
    for tracker_obj in active_trackers: # Renombrado para evitar conflicto
        if tracker_obj.estado == 'Tentativo' and tracker_obj.hits >= MIN_HITS_TO_CONFIRM:
            tracker_obj.estado = 'Confirmado'
            if tracker_obj.display_id is None:
                tracker_obj.display_id = next_sequential_confirmed_id
                next_sequential_confirmed_id +=1
            
            ### MODIFICADO ### La actualización de ultimas_5_ids_confirmadas debe estar aquí
            if tracker_obj.display_id not in ultimas_5_ids_confirmadas : # Evitar duplicados si ya estaba
                ultimas_5_ids_confirmadas.append(tracker_obj.display_id)
                if len(ultimas_5_ids_confirmadas) > MAX_DISPLAY_IDS:
                    ultimas_5_ids_confirmadas.pop(0)

            historial_confirmadas.append({
                'internal_id': tracker_obj.id,
                'tacha_id_secuencial': tracker_obj.display_id,
                'centro_transformado': tracker_obj.centro_transformado.copy(),
                'centro_img': tracker_obj.centro_img.copy(),
                'frame': tracker_obj.frame_idx,
                'clase': 'tacha'
            })
        
        if tracker_obj.misses < MAX_MISSES_TO_DELETE:
            final_trackers.append(tracker_obj)
        else:
            if tracker_obj.display_id: # Solo imprimir si tiene display_id (si fue confirmado alguna vez)
                print(f"Eliminando tracker ID {tracker_obj.id}, ID secuencial {tracker_obj.display_id} por exceso de misses.")
            else:
                print(f"Eliminando tracker tentativo ID {tracker_obj.id} por exceso de misses.")
    active_trackers = final_trackers
    
    # --- ### NUEVO ### INICIO DE LÓGICA PARA DETECTAR TACHAS FALTANTES EN TIEMPO REAL ---
    current_confirmed_active_studs = [t for t in active_trackers if t.estado == 'Confirmado']
    current_confirmed_active_studs.sort(key=lambda t: (t.centro_transformado[1], t.centro_transformado[0]))

    newly_inferred_missing_alerts_this_frame = []

    if len(current_confirmed_active_studs) >= MIN_STUDS_FOR_REALTIME_PATTERN:
        spacings_current_active = []
        for i in range(len(current_confirmed_active_studs) - 1):
            dist = np.linalg.norm(current_confirmed_active_studs[i+1].centro_transformado - 
                                  current_confirmed_active_studs[i].centro_transformado)
            if dist > REALTIME_MIN_SPACING_BEV:
                 spacings_current_active.append(dist)
        
        effective_expected_spacing_bev = REALTIME_EXPECTED_SPACING_BEV
        if spacings_current_active:
            avg_dynamic_spacing = np.mean(spacings_current_active)
            if REALTIME_EXPECTED_SPACING_BEV * 0.5 < avg_dynamic_spacing < REALTIME_EXPECTED_SPACING_BEV * 1.5:
                effective_expected_spacing_bev = avg_dynamic_spacing

        for i in range(len(current_confirmed_active_studs) - 1):
            stud1 = current_confirmed_active_studs[i]
            stud2 = current_confirmed_active_studs[i+1]
            dist_entre_detectadas_bev = np.linalg.norm(stud2.centro_transformado - stud1.centro_transformado)

            if dist_entre_detectadas_bev > (effective_expected_spacing_bev * REALTIME_SPACING_TOLERANCE_FACTOR):
                num_faltantes_estimadas = int(round(dist_entre_detectadas_bev / effective_expected_spacing_bev)) - 1
                if num_faltantes_estimadas > 0:
                    vector_direccion = (stud2.centro_transformado - stud1.centro_transformado) / (num_faltantes_estimadas + 1)
                    for k_idx in range(1, num_faltantes_estimadas + 1): # Renombrado k a k_idx
                        pos_faltante_estimada_bev = stud1.centro_transformado + k_idx * vector_direccion
                        
                        if not (0 <= pos_faltante_estimada_bev[0] < ancho_transformado and \
                                0 <= pos_faltante_estimada_bev[1] < alto_transformado):
                            continue

                        es_duplicada_reciente = False
                        for alert in active_missing_stud_notifications:
                            if np.linalg.norm(alert['bev_pos'] - pos_faltante_estimada_bev) < effective_expected_spacing_bev / 3.0:
                                if frame_idx - alert['creation_frame'] < MAX_ALERT_DURATION_FRAMES / 2 :
                                    es_duplicada_reciente = True
                                    break
                        if es_duplicada_reciente:
                            continue
                        
                        newly_inferred_missing_alerts_this_frame.append({
                            'bev_pos': pos_faltante_estimada_bev,
                            'creation_frame': frame_idx
                        })

    active_missing_stud_notifications.extend(newly_inferred_missing_alerts_this_frame)
    active_missing_stud_notifications = [
        alert for alert in active_missing_stud_notifications
        if (frame_idx - alert['creation_frame']) < MAX_ALERT_DURATION_FRAMES
    ]
    # --- ### FIN ### DE LÓGICA PARA DETECTAR TACHAS FALTANTES EN TIEMPO REAL ---

    # 6. DIBUJAR RESULTADOS
    cv2.rectangle(frame_corregido, (zona_x_inicio, zona_y_inicio), (zona_x_fin, zona_y_fin), (255, 255, 0), 2)
    for tracker_obj in active_trackers: # Renombrado para evitar conflicto
        if tracker_obj.estado == 'Confirmado':
            center_x_img = int(tracker_obj.centro_img[0])
            center_y_img = int(tracker_obj.centro_img[1])
            cv2.circle(frame_corregido, (center_x_img, center_y_img), 7, (0, 255, 0), -1)
            id_text_to_show = tracker_obj.display_id if tracker_obj.display_id is not None else tracker_obj.id
            cv2.putText(frame_corregido, f"ID:{id_text_to_show}",
                        (center_x_img + 10, center_y_img - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- ### NUEVO ### Dibujar Alerta General de Tachas Faltantes (si hay) ---
    if active_missing_stud_notifications:
        alert_text = "Alerta: Posible(s) tacha(s) faltante(s) en ROI"
        (text_w, text_h), baseline = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame_corregido, (5, 5), (10 + text_w + 5, 5 + text_h + baseline + 5), (0,0,0), -1)
        cv2.putText(frame_corregido, alert_text,
                    (10, 5 + text_h + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # --- Dibujar 5 IDs confirmadas ---
    display_start_y = frame_corregido.shape[0] - 20
    id_font_scale = 0.5
    id_font_thickness = 1
    id_font_color = (0, 255, 255) # Amarillo
    for i, confirmed_id in enumerate(reversed(ultimas_5_ids_confirmadas)):
        text_to_display = f"ID Reciente: {confirmed_id}"
        (text_width_id, text_height_id),_ = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, id_font_thickness)
        text_x = frame_corregido.shape[1] - text_width_id - 10
        text_y = display_start_y - i * (text_height_id + 10)
        if text_y - text_height_id < 0:
            break
        cv2.putText(frame_corregido, text_to_display, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, id_font_color, id_font_thickness, cv2.LINE_AA)
    
    # --- FIN DEL NUEVO FLUJO DE TRACKING --- (Este comentario parece fuera de lugar aquí, pertenece a antes de la Odometría Visual)

    # --- Odometría Visual (sin cambios) ---
    # ... (tu código de odometría existente) ...
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
                    _, R_cv, t_cv, mask_rp = cv2.recoverPose(E, dst_pts, src_pts, K) # Renombrado R, t para evitar conflicto
                    escala_asumida = 1.0
                    t_f += escala_asumida * (R_f @ t_cv)
                    R_f = R_cv @ R_f
                    posiciones_vehiculo.append(np.array([t_f[0][0], t_f[2][0]]))
        except cv2.error as e:
            # print(f"Error en Odometría Visual: {e}") # Opcional: para depurar errores de recoverPose
            pass
    prev_gray, prev_kp, prev_des = gray, kp, des

    # --- Dibujo del Mini-mapa (sin cambios) ---
    # ... (tu código de minimapa existente) ...
    if gps_data is not None and len(gps_tiempos) > 0 and len(gps_trayectoria) > 0:
        idx_gps = (np.abs(gps_tiempos - tiempo_actual)).argmin()
        if idx_gps < len(gps_trayectoria):
             gps_posiciones_vehiculo.append(gps_trayectoria[idx_gps])

    mini_mapa_h, mini_mapa_w = 200, 200
    mini_mapa = np.ones((mini_mapa_h, mini_mapa_w, 3), dtype=np.uint8) * 255
    escala_mapa_vo = 5
    centro_mapa_x, centro_mapa_y = mini_mapa_w // 2, mini_mapa_h // 2
    if len(posiciones_vehiculo) > 1:
        for i_mapa in range(1, len(posiciones_vehiculo)): # Renombrado i para evitar conflicto
            x1_vo, z1_vo = posiciones_vehiculo[i_mapa-1]
            x2_vo, z2_vo = posiciones_vehiculo[i_mapa]
            p1_mapa = (int(x1_vo * escala_mapa_vo + centro_mapa_x), int(z1_vo * escala_mapa_vo + centro_mapa_y))
            p2_mapa = (int(x2_vo * escala_mapa_vo + centro_mapa_x), int(z2_vo * escala_mapa_vo + centro_mapa_y))
            if (0 <= p1_mapa[0] < mini_mapa_w and 0 <= p1_mapa[1] < mini_mapa_h and
                0 <= p2_mapa[0] < mini_mapa_w and 0 <= p2_mapa[1] < mini_mapa_h):
                cv2.line(mini_mapa, p1_mapa, p2_mapa, (0, 0, 255), 1)
    if frame_corregido.shape[0] >= mini_mapa_h + 10 and frame_corregido.shape[1] >= mini_mapa_w + 10:
        frame_corregido[10:mini_mapa_h+10, frame_corregido.shape[1]-mini_mapa_w-10:frame_corregido.shape[1]-10] = mini_mapa
    
    # --- Salida de Video y Visualización ---
    out.write(frame_corregido)
    cv2.imshow('Procesamiento en tiempo real', cv2.resize(frame_corregido, (0, 0), fx=0.5, fy=0.5)) 
    if cv2.waitKey(1) & 0xFF == ord('q'): break

    if frame_idx % 100 == 0: print(f"Frame {frame_idx} procesado...")
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# --- ANÁLISIS DE PATRÓN DE COLOCACIÓN DE TACHAS ---
# ... (El resto de tu código de análisis y guardado en Excel permanece igual) ...
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
    for i_dist in range(1, len(historial_tachas_ordenado)): # Renombrado i para evitar conflicto
        tacha_anterior = historial_tachas_ordenado[i_dist-1]
        tacha_actual = historial_tachas_ordenado[i_dist]
        distancia = np.linalg.norm(tacha_actual['centro_transformado'] - tacha_anterior['centro_transformado'])
        distancias_entre_tachas_detectadas_list.append(distancia)
        distancias_info_for_excel.append({
            'ID Tacha Anterior': tacha_anterior.get('tacha_id_secuencial', tacha_anterior.get('id')),
            'ID Tacha Actual': tacha_actual.get('tacha_id_secuencial', tacha_actual.get('id')),
            'Coord. Anterior (X_bev,Y_bev)': f"{tacha_anterior['centro_transformado'][0]:.2f}, {tacha_anterior['centro_transformado'][1]:.2f}",
            'Coord. Actual (X_bev,Y_bev)': f"{tacha_actual['centro_transformado'][0]:.2f}, {tacha_actual['centro_transformado'][1]:.2f}",
            'Distancia BEV (pix)': f"{distancia:.2f}"
        })
else:
    print("No hay suficientes tachas detectadas (se necesitan al menos 2) para calcular distancias.")
    summary_metrics_for_excel.append({"Métrica": "Cálculo de distancias", "Valor": "Insuficientes tachas (requiere >= 2)."})

media_espaciado_pix_bev = 0
std_dev_espaciado_pix_bev = 0
if distancias_entre_tachas_detectadas_list:
    media_espaciado_pix_bev = np.mean(distancias_entre_tachas_detectadas_list)
    std_dev_espaciado_pix_bev = np.std(distancias_entre_tachas_detectadas_list)
    min_espaciado_pix_bev = np.min(distancias_entre_tachas_detectadas_list)
    max_espaciado_pix_bev = np.max(distancias_entre_tachas_detectadas_list)
    print(f"\nMedia del espaciado (detectadas): {media_espaciado_pix_bev:.2f} píxeles (BEV)")
    print(f"Desviación estándar del espaciado (detectadas): {std_dev_espaciado_pix_bev:.2f} píxeles (BEV)")
    summary_metrics_for_excel.append({"Métrica": "Media espaciado (píxeles BEV)", "Valor": f"{media_espaciado_pix_bev:.2f}"})
    summary_metrics_for_excel.append({"Métrica": "Std Dev espaciado (píxeles BEV)", "Valor": f"{std_dev_espaciado_pix_bev:.2f}"})
    summary_metrics_for_excel.append({"Métrica": "Min espaciado (píxeles BEV)", "Valor": f"{min_espaciado_pix_bev:.2f}"})
    summary_metrics_for_excel.append({"Métrica": "Max espaciado (píxeles BEV)", "Valor": f"{max_espaciado_pix_bev:.2f}"})

tachas_faltantes_estimadas = []
if media_espaciado_pix_bev > 0 and historial_tachas_ordenado:
    print(f"\n--- Estimando Tachas Faltantes (usando media espaciado: {media_espaciado_pix_bev:.2f}) ---")
    tachas_faltantes_estimadas = detectar_tachas_faltantes(historial_tachas_ordenado, media_espaciado_pix_bev)
    print(f"Se estimaron {len(tachas_faltantes_estimadas)} tachas faltantes.")
    summary_metrics_for_excel.append({"Métrica": "Número de tachas faltantes estimadas (post-proceso)", "Valor": len(tachas_faltantes_estimadas)})
else:
    print("No se pudo estimar tachas faltantes (post-proceso).")
    summary_metrics_for_excel.append({"Métrica": "Estimación tachas faltantes (post-proceso)", "Valor": "No realizada o 0."})

df_sheet1_summary = pd.DataFrame(summary_metrics_for_excel)
df_sheet1_distancias_detalle = pd.DataFrame(distancias_info_for_excel)

inventario_data_for_excel = []
for tacha_detectada in historial_tachas_ordenado:
    inventario_data_for_excel.append({
        'ID_Tacha': tacha_detectada.get('tacha_id_secuencial', tacha_detectada.get('id')),
        'Centro_Transformado_X': f"{tacha_detectada['centro_transformado'][0]:.2f}",
        'Centro_Transformado_Y': f"{tacha_detectada['centro_transformado'][1]:.2f}",
        'Frame_Referencia': tacha_detectada['frame'],
        'Estado': 'Detectada'
    })
for tacha_faltante in tachas_faltantes_estimadas:
    inventario_data_for_excel.append({
        'ID_Tacha': tacha_faltante['tacha_id_secuencial'],
        'Centro_Transformado_X': f"{tacha_faltante['centro_transformado'][0]:.2f}",
        'Centro_Transformado_Y': f"{tacha_faltante['centro_transformado'][1]:.2f}",
        'Frame_Referencia': tacha_faltante['frame'],
        'Estado': tacha_faltante['Estado']
    })

df_sheet2_inventario_tachas = pd.DataFrame(inventario_data_for_excel)
if not df_sheet2_inventario_tachas.empty:
    df_sheet2_inventario_tachas['Centro_Transformado_Y_float'] = pd.to_numeric(df_sheet2_inventario_tachas['Centro_Transformado_Y'], errors='coerce')
    df_sheet2_inventario_tachas['Centro_Transformado_X_float'] = pd.to_numeric(df_sheet2_inventario_tachas['Centro_Transformado_X'], errors='coerce')
    df_sheet2_inventario_tachas = df_sheet2_inventario_tachas.sort_values(by=['Centro_Transformado_Y_float', 'Centro_Transformado_X_float']).drop(columns=['Centro_Transformado_Y_float', 'Centro_Transformado_X_float'])

log_confirmadas_for_excel = []
for tracker_info in historial_confirmadas:
    log_confirmadas_for_excel.append({
        'Internal_Tracker_ID': tracker_info['internal_id'],
        'Tacha_ID_Secuencial': tracker_info['tacha_id_secuencial'],
        'Centro_Transformado_X': f"{tracker_info['centro_transformado'][0]:.2f}",
        'Centro_Transformado_Y': f"{tracker_info['centro_transformado'][1]:.2f}",
        'Frame_Confirmacion': tracker_info['frame'],
        'Clase_Detectada': tracker_info['clase'],
        'Estado_Tracker': 'Confirmado' 
    })
df_sheet3_log_confirmadas = pd.DataFrame(log_confirmadas_for_excel)

fecha_actual_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
nombre_archivo_excel = f"{fecha_actual_str}_Analisis_Completo_Tachas_vRT.xlsx" # Modificado nombre

try:
    with pd.ExcelWriter(nombre_archivo_excel, engine='openpyxl') as writer:
        df_sheet1_summary.to_excel(writer, sheet_name='Resumen y Distancias', index=False, header=True)
        if not df_sheet1_distancias_detalle.empty:
            writer.sheets['Resumen y Distancias'].cell(row=len(df_sheet1_summary) + 3, column=1, value="Detalle de Distancias entre Tachas Detectadas")
            df_sheet1_distancias_detalle.to_excel(writer, sheet_name='Resumen y Distancias', index=False, startrow=len(df_sheet1_summary) + 4)
        
        df_sheet2_inventario_tachas.to_excel(writer, sheet_name='Inventario Tachas (Det_Falt)', index=False)
        df_sheet3_log_confirmadas.to_excel(writer, sheet_name='Log Tachas Confirmadas', index=False)
    print(f"\nInforme de análisis completo guardado en: '{nombre_archivo_excel}'")
except Exception as e:
    print(f"\nError al guardar el informe de análisis en Excel: {e}")


if posiciones_vehiculo and (gps_data is not None and len(gps_posiciones_vehiculo) > 0):
    odo_np = np.array(posiciones_vehiculo)
    gps_np = np.array(gps_posiciones_vehiculo)
    plt.figure(figsize=(10, 8))
    plt.plot(odo_np[:, 0], odo_np[:, 1], 'g-', label='Odometría Visual (Escala Relativa)')
    if gps_np.ndim == 2 and gps_np.shape[1] == 2:
        plt.plot(gps_np[:, 1], gps_np[:, 0], 'b--', label='GPS (Longitud, Latitud)')
    else:
        print("Datos GPS no tienen el formato esperado para graficar.")
    plt.title('Comparación de Trayectorias (VO y GPS)')
    plt.xlabel('X / Longitud')
    plt.ylabel('Z / Latitud')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    try:
        plt.savefig('trayectoria_comparativa_vo_gps_vRT.png', dpi=300) # Modificado nombre
        print("Gráfico de trayectoria comparativa guardado.")
    except Exception as e:
        print(f"Error al guardar el gráfico de trayectoria: {e}")
else:
    print("No hay suficientes datos de odometría o GPS para graficar la trayectoria comparativa.")

print("\nProcesamiento finalizado.")