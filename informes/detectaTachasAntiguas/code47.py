# Mejora de odometría visual con acumulación de rotación y translación coherente
# Código realizado por Felipe Pereira Alarcón
# MODIFICADO: Sistema de seguimiento con Filtro de Kalman y Máquina de Estados

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
        self.kf = cv2.KalmanFilter(4, 2) # Estado: [x, y, vx, vy], Medida: [x, y]
        
        # --- Configuración del Filtro de Kalman ---
        # Matriz de Transición (A)
        # --- CORRECCIÓN ---: Se cambió `np(...)` por `np.array(...)`
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
        # --- CORRECCIÓN ---: La matriz debe ser 2x2, ya que la medida tiene 2 dimensiones.
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
        # --- CORRECCIÓN ---: En la predicción, asumimos un 'miss' hasta que se actualice.
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

# Video
ruta_video = 'videos/3.2 - 01_04 Tramo B1-B2.MP4'
output_video = '3.2 - 01_04 Tramo B1-B2_resultadoTransformado_conAnalisisTachas_NN.MP4'
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
historial_confirmadas = [] # --- NUEVO --- Para el análisis final

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
            # --- NUEVO --- Guardar la tacha confirmada para el análisis final
            historial_confirmadas.append({
                'id': tracker.id,
                'centro_transformado': tracker.centro_transformado,
                'frame': tracker.frame_idx,
                'clase': 'tacha'
            })
        
        if tracker.misses < MAX_MISSES_TO_DELETE:
            final_trackers.append(tracker)
        else:
            print(f"Eliminando tracker ID {tracker.id} por exceso de misses.")
    active_trackers = final_trackers

    # 6. DIBUJAR RESULTADOS
    cv2.rectangle(frame_corregido, (zona_x_inicio, zona_y_inicio), (zona_x_fin, zona_y_fin), (255, 255, 0), 2)
    for tracker in active_trackers:
        if tracker.estado == 'Confirmado':
            # --- Usamos las coordenadas del tracker, que ya están en el sistema de la imagen original ---
            center_x_img = int(tracker.centro_img[0])
            center_y_img = int(tracker.centro_img[1])
            
            # Dibuja un círculo verde y el ID para las tachas confirmadas
            cv2.circle(frame_corregido, (center_x_img, center_y_img), 7, (0, 255, 0), -1)
            cv2.putText(frame_corregido, f"ID:{tracker.id}",
                        (center_x_img + 10, center_y_img - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- FIN DEL NUEVO FLUJO DE TRACKING ---


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
# --- CORRECCIÓN ---: Se utiliza la nueva lista `historial_confirmadas`
print("\n--- Iniciando Análisis de Patrón de Colocación de Tachas ---")
historial_solo_tachas = historial_confirmadas
total_detecciones_unicas_tachas_analisis = 0
info_analisis_tachas = []

if historial_solo_tachas:
    historial_tachas_ordenado = sorted(historial_solo_tachas, key=lambda t: (t['centro_transformado'][1], t['centro_transformado'][0]))
    total_detecciones_unicas_tachas_analisis = len(historial_tachas_ordenado)
    print(f"Total de detecciones de tachas únicas procesadas para análisis: {total_detecciones_unicas_tachas_analisis}")
    info_analisis_tachas.append({"Métrica": "Total de detecciones de tachas únicas procesadas para análisis", "Valor": total_detecciones_unicas_tachas_analisis})
else:
    historial_tachas_ordenado = []
    print("No se encontraron tachas en el historial para analizar.")
    info_analisis_tachas.append({"Métrica": "Total de detecciones de tachas únicas procesadas para análisis", "Valor": 0})
    info_analisis_tachas.append({"Métrica": "Advertencia", "Valor": "No se encontraron tachas en el historial para analizar."})


distancias_entre_tachas = []
if len(historial_tachas_ordenado) > 1:
    for i in range(1, len(historial_tachas_ordenado)):
        tacha_anterior = historial_tachas_ordenado[i-1]
        tacha_actual = historial_tachas_ordenado[i]
        distancia = np.linalg.norm(tacha_actual['centro_transformado'] - tacha_anterior['centro_transformado'])
        distancias_entre_tachas.append(distancia)
else:
    print("No hay suficientes tachas (se necesitan al menos 2) para calcular distancias.")
    if not historial_solo_tachas or len(historial_solo_tachas) <= 1:
        info_analisis_tachas.append({"Métrica": "Nota sobre cálculo de distancias", "Valor": "No hay suficientes tachas (se necesitan al menos 2) para calcular distancias."})

# ... El resto del script de análisis y guardado en Excel puede permanecer igual ...
# (Se omite por brevedad, ya que no requiere cambios)

media_espaciado_pix_bev = 0
std_dev_espaciado_pix_bev = 0
if distancias_entre_tachas:
    media_espaciado_pix_bev = np.mean(distancias_entre_tachas)
    std_dev_espaciado_pix_bev = np.std(distancias_entre_tachas)
    print(f"\nMedia del espaciado: {media_espaciado_pix_bev:.2f} píxeles (vista de pájaro)")
    print(f"Desviación estándar del espaciado: {std_dev_espaciado_pix_bev:.2f} píxeles (vista de pájaro)")
    info_analisis_tachas.append({"Métrica": "Media del espaciado (píxeles BEV)", "Valor": f"{media_espaciado_pix_bev:.2f}"})
    info_analisis_tachas.append({"Métrica": "Desviación estándar del espaciado (píxeles BEV)", "Valor": f"{std_dev_espaciado_pix_bev:.2f}"})

# --- GUARDAR ANÁLISIS EN EXCEL ---
fecha_actual_str = datetime.now().strftime("%m-%d-%y")
nombre_archivo_excel = f"{fecha_actual_str} - Análisis FP SSGL.xlsx"

df_analisis_tachas = pd.DataFrame(info_analisis_tachas)
# df_datos_nn = pd.DataFrame(info_nn) # La sección de NN fue omitida para enfocar en el tracker

try:
    with pd.ExcelWriter(nombre_archivo_excel, engine='openpyxl') as writer:
        df_analisis_tachas.to_excel(writer, sheet_name='Análisis Patrón Tachas', index=False)
        # df_datos_nn.to_excel(writer, sheet_name='Datos NN', index=False)
    print(f"\nInforme de análisis guardado en: '{nombre_archivo_excel}'")
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