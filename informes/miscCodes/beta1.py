# Mejora de odometría visual con acumulación de rotación y translación coherente
# Código realizado por Felipe Pereira Alarcón
# MODIFICADO: Sistema de seguimiento con Filtro de Kalman y Máquina de Estados
# MODIFICADO: Detección de 'Tacha Faltante' en tiempo real.
# MODIFICADO: Cálculo de distancia entre tachas confirmadas en Log Excel.

import cv2
import numpy as np
import torch
import statistics
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

        # --- Configuración del Filtro de Kalman ---
        # Matriz de Transición (A)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],   # x_new = x_old + vx
                                              [0, 1, 0, 1],   # y_new = y_old + vy
                                              [0, 0, 1, 0],   # vx_new = vx_old
                                              [0, 0, 0, 1]], np.float32)

        # Matriz de Medida (H)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],  # Medimos solo la posición x
                                               [0, 1, 0, 0]], np.float32) # Medimos solo la posición y

        # Ruido del Proceso (Q)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Ruido de la Medida (R)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        # Estado inicial [x, y, vx, vy]
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
        """Predice la nueva posición y actualiza contadores."""
        self.age += 1
        self.misses += 1
        return self.kf.predict()

    def update(self, centro_img, centro_transformado):
        """Actualiza el tracker con una nueva detección."""
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
# Nuevos parámetros para filtrado geométrico de tachas
MIN_TACHA_WIDTH_PX = 5
MAX_TACHA_WIDTH_PX = 80
MIN_TACHA_HEIGHT_PX = 5
MAX_TACHA_HEIGHT_PX = 80
MIN_TACHA_ASPECT_RATIO = 0.3
MAX_TACHA_ASPECT_RATIO = 2.5

# --- NUEVO --- Parámetros para el tracker (puedes ajustarlos)
MIN_HITS_TO_CONFIRM = 3
MAX_MISSES_TO_DELETE = 5

# --- NUEVO --- Parámetros para detección de Tacha Faltante en Tiempo Real
min_frames_entre_detecciones = 5 # Mínimo de frames antes de registrar una nueva detección para el cálculo de intervalo
ventana_inicial = 5              # Número de detecciones para empezar a calcular promedio y desv. est.
tolerancia_frames = 2.0          # Multiplicador de desv. est. para el umbral de 'falta'

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
        if height_b == 0: continue
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

# Actualización tachas

def detectar_tachas_faltantes(tachas_ordenadas, media_espaciado_bev, umbral_multiplicador=1.75):
    """
    Detecta tachas faltantes basándose en el espaciado medio de las tachas detectadas.

    Args:
        tachas_ordenadas (list): Lista de diccionarios de tachas confirmadas, ordenadas por su posición.
        media_espaciado_bev (float): El espaciado medio entre tachas en la vista de pájaro (BEV).
        umbral_multiplicador (float): Multiplicador para determinar si un espacio es suficientemente grande
                                      como para inferir una tacha faltante (ej. 1.75 veces la media).

    Returns:
        list: Lista de diccionarios, donde cada diccionario representa una tacha faltante estimada.
    """
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
                        'frame': tacha_actual['frame'], # Frame of the preceding detected stud as reference
                        'clase': 'tacha_faltante_estimada',
                        'Estado': 'Faltante' # For the inventory sheet
                    })
    return faltantes

# Video
ruta_video = 'videos/3.2 - 01_04 Tramo B1-B2.MP4'
output_video = '3.2 - 01_04 Tramo B1-B2_resultadoTransformado_conAnalisisTachas_NN_FaltaTacha_v2.MP4' # Nombre de salida modificado
cap = cv2.VideoCapture(ruta_video)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el video {ruta_video}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# --- Definición de Zona de Interés (ROI) y Transformación de Perspectiva ---
zona_x_inicio = int(frame_width * 0.20)
zona_x_fin = int(frame_width * 0.50)
zona_y_inicio = int(frame_height * 0.55)
zona_y_fin = frame_height
pts_origen = np.float32([[zona_x_inicio, zona_y_inicio], [zona_x_fin, zona_y_inicio], [zona_x_inicio, zona_y_fin], [zona_x_fin, zona_y_fin]])
ancho_transformado, alto_transformado = 400, 600
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

# --- NUEVO --- Variables para Tacha Faltante en Tiempo Real
detecciones_frames = []
promedio_frames = 0.0
desviacion_frames = 0.0

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
    unmatched_detections = detecciones_actuales
    if active_trackers and detecciones_actuales:
        cost_matrix = np.zeros((len(active_trackers), len(detecciones_actuales)))
        for t, tracker in enumerate(active_trackers):
            for d, det in enumerate(detecciones_actuales):
                pred_x, pred_y = tracker.kf.statePost[0], tracker.kf.statePost[1]
                det_x, det_y = det['cx'], det['cy']
                cost_matrix[t, d] = np.linalg.norm([pred_x - det_x, pred_y - det_y])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        max_dist_threshold = 50
        matched_indices = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_dist_threshold:
                tracker = active_trackers[r]
                det = detecciones_actuales[c]

                # 4a. ACTUALIZACIÓN DE TRACKERS ASOCIADOS
                centro_img_det = np.array([det['cx'], det['cy']])
                centro_transformado_det = transformar_punto(centro_img_det, M)
                tracker.update(centro_img_det, centro_transformado_det)
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
    for tracker in active_trackers:
        if tracker.estado == 'Tentativo' and tracker.hits >= MIN_HITS_TO_CONFIRM:
            tracker.estado = 'Confirmado'

            # Nuevo asignador de id secuencial
            if tracker.display_id is None:
                tracker.display_id = next_sequential_confirmed_id
                next_sequential_confirmed_id +=1

                # --- NUEVO: Registrar frame para Tacha Faltante ---
                if not detecciones_frames or (frame_idx - detecciones_frames[-1]) >= min_frames_entre_detecciones:
                    detecciones_frames.append(frame_idx)
                    if len(detecciones_frames) >= ventana_inicial + 1:
                        intervalos = [j - i for i, j in zip(detecciones_frames[:-1], detecciones_frames[1:])]
                        window_intervals = intervalos[-ventana_inicial:]
                        promedio_frames = sum(window_intervals) / len(window_intervals)
                        desviacion_frames = statistics.stdev(window_intervals) if len(window_intervals) > 1 else 0
                        print(f"[DEBUG TACHA FALTANTE] Promedio: {promedio_frames:.2f}, desv: {desviacion_frames:.2f}")
                # --- FIN NUEVO ---

                # Guardar la tacha confirmada para el análisis final (SOLO LA PRIMERA VEZ QUE SE CONFIRMA)
                historial_confirmadas.append({
                    'internal_id': tracker.id,
                    'tacha_id_secuencial': tracker.display_id,
                    'centro_transformado': tracker.centro_transformado.copy(),
                    'centro_img': tracker.centro_img.copy(),
                    'frame': tracker.frame_idx,
                    'clase': 'tacha'
                })

                # Actualizar lista de Displays
                ultimas_5_ids_confirmadas.append(tracker.display_id)
                if len(ultimas_5_ids_confirmadas) > MAX_DISPLAY_IDS:
                    ultimas_5_ids_confirmadas.pop(0)


        if tracker.misses < MAX_MISSES_TO_DELETE:
            final_trackers.append(tracker)
        else:
            print(f"Eliminando tracker ID {tracker.id}, ID secuencial {tracker.display_id} por exceso de misses.")
    active_trackers = final_trackers


    # --- NUEVO: Lógica de Alerta de Tacha Faltante ---
    if promedio_frames > 0 and len(detecciones_frames) > ventana_inicial:
        tiempo_desde_ultima = frame_idx - detecciones_frames[-1]
        umbral_falta = promedio_frames + tolerancia_frames * desviacion_frames
        if tiempo_desde_ultima > umbral_falta:
            print(f"[ALERTA] Posible falta de tacha en frame {frame_idx} (Tiempo: {tiempo_desde_ultima}, Umbral: {umbral_falta:.2f})")
            cv2.putText(frame_corregido, "FALTA TACHA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    # --- FIN NUEVO ---


    # 6. DIBUJAR RESULTADOS
    cv2.rectangle(frame_corregido, (zona_x_inicio, zona_y_inicio), (zona_x_fin, zona_y_fin), (255, 255, 0), 2)
    for tracker in active_trackers:
        if tracker.estado == 'Confirmado':
            center_x_img = int(tracker.centro_img[0])
            center_y_img = int(tracker.centro_img[1])

            cv2.circle(frame_corregido, (center_x_img, center_y_img), 7, (0, 255, 0), -1)
            id_text_to_show = tracker.display_id if tracker.display_id is not None else tracker.id
            cv2.putText(frame_corregido, f"ID:{id_text_to_show}",
                        (center_x_img + 10, center_y_img - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # --- Odometría Visual (sin cambios) ---
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
                    _, R, t, mask_rp = cv2.recoverPose(E, dst_pts, src_pts, K)
                    escala_asumida = 1.0
                    t_f += escala_asumida * (R_f @ t)
                    R_f = R @ R_f
                    posiciones_vehiculo.append(np.array([t_f[0][0], t_f[2][0]]))
        except cv2.error as e:
            pass
    prev_gray, prev_kp, prev_des = gray, kp, des

    # --- Dibujo del Mini-mapa (sin cambios) ---
    if gps_data is not None and len(gps_tiempos) > 0 and len(gps_trayectoria) > 0:
        idx_gps = (np.abs(gps_tiempos - tiempo_actual)).argmin()
        if idx_gps < len(gps_trayectoria):
             gps_posiciones_vehiculo.append(gps_trayectoria[idx_gps])

    mini_mapa_h, mini_mapa_w = 200, 200
    mini_mapa = np.ones((mini_mapa_h, mini_mapa_w, 3), dtype=np.uint8) * 255
    escala_mapa_vo = 5
    centro_mapa_x, centro_mapa_y = mini_mapa_w // 2, mini_mapa_h // 2
    if len(posiciones_vehiculo) > 1:
        for i in range(1, len(posiciones_vehiculo)):
            x1_vo, z1_vo = posiciones_vehiculo[i-1]
            x2_vo, z2_vo = posiciones_vehiculo[i]
            p1_mapa = (int(x1_vo * escala_mapa_vo + centro_mapa_x), int(z1_vo * escala_mapa_vo + centro_mapa_y))
            p2_mapa = (int(x2_vo * escala_mapa_vo + centro_mapa_x), int(z2_vo * escala_mapa_vo + centro_mapa_y))
            if (0 <= p1_mapa[0] < mini_mapa_w and 0 <= p1_mapa[1] < mini_mapa_h and
                0 <= p2_mapa[0] < mini_mapa_w and 0 <= p2_mapa[1] < mini_mapa_h):
                cv2.line(mini_mapa, p1_mapa, p2_mapa, (0, 0, 255), 1)
    if frame_corregido.shape[0] >= mini_mapa_h + 10 and frame_corregido.shape[1] >= mini_mapa_w + 10:
        frame_corregido[10:mini_mapa_h+10, frame_corregido.shape[1]-mini_mapa_w-10:frame_corregido.shape[1]-10] = mini_mapa

    # Dibujar 5 IDs confirmadas
    display_start_y = frame_corregido.shape[0] - 20
    id_font_scale = 0.5
    id_font_thickness = 1
    id_font_color = (0, 255, 255)

    for i, confirmed_id in enumerate(reversed(ultimas_5_ids_confirmadas)):
        text_to_display = f"ID Reciente: {confirmed_id}"
        (text_width, text_height),baseline = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, id_font_thickness)
        text_x = frame_corregido.shape[1] - text_width - 10
        text_y = display_start_y - i * (text_height + 10)

        if text_y - text_height < 0:
            break

        cv2.putText(frame_corregido, text_to_display, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, id_font_color, id_font_thickness, cv2.LINE_AA)

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
print("\n--- Iniciando Análisis de Patrón de Colocación de Tachas ---")
historial_solo_tachas = historial_confirmadas
total_detecciones_unicas_tachas_analisis = 0
summary_metrics_for_excel = [] # Replaces info_analisis_tachas for the summary part

if historial_solo_tachas:
    # Sort by Y then X in bird's-eye view for sequential analysis
    historial_tachas_ordenado = sorted(historial_solo_tachas, key=lambda t: (t['centro_transformado'][1], t['centro_transformado'][0]))
    total_detecciones_unicas_tachas_analisis = len(historial_tachas_ordenado)
    print(f"Total de detecciones de tachas únicas procesadas para análisis: {total_detecciones_unicas_tachas_analisis}")
    summary_metrics_for_excel.append({"Métrica": "Total de tachas únicas detectadas", "Valor": total_detecciones_unicas_tachas_analisis})
else:
    historial_tachas_ordenado = []
    print("No se encontraron tachas en el historial para analizar.")
    summary_metrics_for_excel.append({"Métrica": "Total de tachas únicas detectadas", "Valor": 0})
    summary_metrics_for_excel.append({"Métrica": "Advertencia", "Valor": "No se encontraron tachas para análisis."})

distancias_entre_tachas_detectadas_list = [] # For storing actual distance values
distancias_info_for_excel = [] # For storing dicts for DataFrame (Sheet 1 detail)

if len(historial_tachas_ordenado) > 1:
    for i in range(1, len(historial_tachas_ordenado)):
        tacha_anterior = historial_tachas_ordenado[i-1]
        tacha_actual = historial_tachas_ordenado[i]
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

# --- Detección de Tachas Faltantes ---
tachas_faltantes_estimadas = []
if media_espaciado_pix_bev > 0 and historial_tachas_ordenado:
    print(f"\n--- Estimando Tachas Faltantes (usando media espaciado: {media_espaciado_pix_bev:.2f}) ---")
    tachas_faltantes_estimadas = detectar_tachas_faltantes(historial_tachas_ordenado, media_espaciado_pix_bev)
    print(f"Se estimaron {len(tachas_faltantes_estimadas)} tachas faltantes.")
    summary_metrics_for_excel.append({"Métrica": "Número de tachas faltantes estimadas", "Valor": len(tachas_faltantes_estimadas)})
else:
    print("No se pudo estimar tachas faltantes (media de espaciado no válida o no hay suficientes tachas detectadas).")
    summary_metrics_for_excel.append({"Métrica": "Estimación tachas faltantes", "Valor": "No realizada o 0."})

# --- Preparación de datos para Excel ---

# Hoja 1: Análisis de Distancias y Resumen
df_sheet1_summary = pd.DataFrame(summary_metrics_for_excel)
df_sheet1_distancias_detalle = pd.DataFrame(distancias_info_for_excel)

# Hoja 2: Inventario de Tachas (Detectadas y Faltantes)
inventario_data_for_excel = []
for tacha_detectada in historial_tachas_ordenado:
    inventario_data_for_excel.append({
        'ID_Tacha': tacha_detectada.get('tacha_id_secuencial', tacha_detectada.get('id')),
        'Centro_Transformado_X': f"{tacha_detectada['centro_transformado'][0]:.2f}",
        'Centro_Transformado_Y': f"{tacha_detectada['centro_transformado'][1]:.2f}",
        'Frame_Referencia': tacha_detectada['frame'],
        'Estado': 'Detectada'
    })
for tacha_faltante in tachas_faltantes_estimadas: # tacha_faltante already has 'Estado':'Faltante'
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

# --- INICIO: MODIFICACIÓN Hoja 3: Log de Tachas Confirmadas (Trackers) ---
log_confirmadas_for_excel = []
# Ordenamos el historial por el ID secuencial para asegurar el cálculo correcto de la distancia
historial_confirmadas_sorted_log = sorted(historial_confirmadas, key=lambda x: x['tacha_id_secuencial'])

for i, tracker_info in enumerate(historial_confirmadas_sorted_log):
    current_data = {
        'Internal_Tracker_ID': tracker_info['internal_id'],
        'Tacha_ID_Secuencial': tracker_info['tacha_id_secuencial'],
        'Centro_Transformado_X': f"{tracker_info['centro_transformado'][0]:.2f}",
        'Centro_Transformado_Y': f"{tracker_info['centro_transformado'][1]:.2f}",
        'Frame_Confirmacion': tracker_info['frame'],
        'Clase_Detectada': tracker_info['clase'],
        'Estado_Tracker': 'Confirmado',
        'ID_Secuencial_Anterior': 'N/A', # Nueva columna
        'Coord_Anterior_X': 'N/A',     # Nueva columna
        'Coord_Anterior_Y': 'N/A',     # Nueva columna
        'Distancia_BEV_pix': 'N/A'     # Nueva columna
    }

    # Si no es la primera tacha, calculamos la distancia con la anterior
    if i > 0:
        prev_tracker_info = historial_confirmadas_sorted_log[i-1]
        current_coords = tracker_info['centro_transformado']
        prev_coords = prev_tracker_info['centro_transformado']
        distancia = np.linalg.norm(current_coords - prev_coords)

        current_data['ID_Secuencial_Anterior'] = prev_tracker_info['tacha_id_secuencial']
        current_data['Coord_Anterior_X'] = f"{prev_coords[0]:.2f}"
        current_data['Coord_Anterior_Y'] = f"{prev_coords[1]:.2f}"
        current_data['Distancia_BEV_pix'] = f"{distancia:.2f}"

    log_confirmadas_for_excel.append(current_data)

df_sheet3_log_confirmadas = pd.DataFrame(log_confirmadas_for_excel)
# --- FIN: MODIFICACIÓN Hoja 3 ---


# --- GUARDAR ANÁLISIS EN EXCEL ---
fecha_actual_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
nombre_archivo_excel = f"{fecha_actual_str}_Analisis_Completo_Tachas_v2.xlsx" # Nombre de salida modificado

try:
    with pd.ExcelWriter(nombre_archivo_excel, engine='openpyxl') as writer:
        df_sheet1_summary.to_excel(writer, sheet_name='Resumen y Distancias', index=False, header=True)
        if not df_sheet1_distancias_detalle.empty:
            writer.sheets['Resumen y Distancias'].cell(row=len(df_sheet1_summary) + 3, column=1, value="Detalle de Distancias entre Tachas Detectadas (Ordenadas por BEV)")
            df_sheet1_distancias_detalle.to_excel(writer, sheet_name='Resumen y Distancias', index=False, startrow=len(df_sheet1_summary) + 4)

        df_sheet2_inventario_tachas.to_excel(writer, sheet_name='Inventario Tachas (Det_Falt)', index=False)
        df_sheet3_log_confirmadas.to_excel(writer, sheet_name='Log Tachas Confirmadas', index=False) # Guardamos la hoja modificada
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
        plt.savefig('trayectoria_comparativa_vo_gps.png', dpi=300)
        print("Gráfico de trayectoria comparativa guardado.")
    except Exception as e:
        print(f"Error al guardar el gráfico de trayectoria: {e}")
else:
    print("No hay suficientes datos de odometría o GPS para graficar la trayectoria comparativa.")

print("\nProcesamiento finalizado.")