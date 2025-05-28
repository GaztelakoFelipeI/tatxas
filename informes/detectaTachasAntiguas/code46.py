# Mejora de odometría visual con acumulación de rotación y translación coherente
# Se ajusta el código original con una estimación más estable
# IMPORTANTE: SE AÑADE ANÁLISIS DE PATRÓN DE COLOCACIÓN DE TACHAS + INTEGRACIÓN CONCEPTUAL DE RED NEURONAL PARA VALIDACIÓN DE PATRONES
# Código realizado por Felipe Pereira Alarcón, Ingeniero Civil Informático, Universidad Andrés Bello, Chile - Practicante SSGL
# Fecha: 2025-22-05
# MODIFICADO: Se añade guardado de resultados de análisis en Excel.

import cv2
import numpy as np
import torch
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from datetime import datetime # Para el nombre del archivo Excel

# Configuración
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print("Usando dispositivo:", device)

model_captafaros = YOLO('captaPT/best.pt').to(device)
model_tachas = YOLO('tachasPT/best.pt').to(device)
model_senaleticas = YOLO('PkPT/best.pt').to(device)

# Parámetros
conf_threshold = 0.3
historial_max_len = 100
tolerancia_frames = 1.5
ventana_inicial = 5
min_frames_entre_detecciones = 40
umbral_espacio = 25
umbral_tiempo = 30

# Nuevos parámetros para filtrado geométrico de tachas (modificar a gusto)
MIN_TACHA_WIDTH_PX = 5
MAX_TACHA_WIDTH_PX = 80
MIN_TACHA_HEIGHT_PX = 5
MAX_TACHA_HEIGHT_PX = 80
MIN_TACHA_ASPECT_RATIO = 0.3
MAX_TACHA_ASPECT_RATIO = 2.5

# Nuevos Parámetros para el Filtro de Persistencia Temporal
MAX_CANDIDATE_AGE_FRAMES = 4
MIN_HITS_FOR_PERSISTENCE = 2
PERSISTENCE_PIXEL_TOLERANCE = 15

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

# Historial detecciones MODIFICADO: ahora objeto_historial guardará todas las detecciones para análisis posterior.
detecciones_frames = []
promedio_frames = None
objeto_historial = []
tacha_id_counter = 0

# Trayectoria GPS
# Asegúrate de que el archivo "metadata/3.2 - 01_04 Tramo B1-B2.xlsx" exista
try:
    gps_data = pd.read_excel("metadata/3.2 - 01_04 Tramo B1-B2.xlsx")
    gps_trayectoria = gps_data[['Latitud', 'Longitud']].values
    gps_tiempos = gps_data['Tiempo'].values
    gps_posiciones_vehiculo = []
except FileNotFoundError:
    print("Archivo GPS 'metadata/3.2 - 01_04 Tramo B1-B2.xlsx' no encontrado. Se omitirá la carga de datos GPS.")
    gps_data = None
    gps_trayectoria = []
    gps_tiempos = []
    gps_posiciones_vehiculo = []


# Funciones auxiliares
def transformar_punto(punto, matriz):
    punto_homog = np.array([[punto[0], punto[1], 1.0]]).T
    transformado = np.dot(matriz, punto_homog)
    if transformado[2] == 0: # Evitar división por cero
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
    #filtrados = []
    filtrados_detailed = []
    for obj in objetos:
        x1b, y1b, x2b, y2b, conf, cls_id = obj
        if conf < conf_threshold:
            continue
        cx, cy = (x1b + x2b) / 2, (y1b + y2b) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        
        # Análisis de tachas nuevas para evitar la confusión con la distancia entre tachas (visual)
        width_b = x2b - x1b
        height_b = y2b - y1b

        if not (MIN_TACHA_WIDTH_PX <= width_b <= MAX_TACHA_WIDTH_PX):
            continue
        if not (MIN_TACHA_HEIGHT_PX <= height_b <= MAX_TACHA_HEIGHT_PX):
            continue

        if height_b == 0:
            continue
        aspect_ratio_b = width_b / height_b
        if not (MIN_TACHA_ASPECT_RATIO <= aspect_ratio_b <= MAX_TACHA_ASPECT_RATIO):
            continue
        
        demasiado_cerca_otros = False
        if len(otros1) > 0:
            for x1o, y1o, x2o, y2o, *_ in otros1:
                cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2
                if np.linalg.norm([cx - cxo, cy - cyo]) < 30:
                    demasiado_cerca_otros = True
                    break
        if demasiado_cerca_otros: continue

        if len(otros2) > 0:
            for x1o, y1o, x2o, y2o, *_ in otros2:
                cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2
                if np.linalg.norm([cx - cxo, cy - cyo]) < 30:
                    demasiado_cerca_otros = True
                    break
        if demasiado_cerca_otros: continue

        filtrados_detailed.append({
            'cx': cx, 'cy': cy,
            'x1b': x1b, 'y1b': y1b, 'x2b': x2b, 'y2b': y2b,
            'frame_idx': current_frame_idx
        })
        
        #filtrados.append(np.array([cx, cy]))
    return filtrados_detailed

def registrar_detecciones_nuevas(centros_img_coords_filtrados, clase_actual, historial_existente, frame_actual, matriz_perspectiva):
    global tacha_id_counter
    nuevas_registradas = []
    for centro_img in centros_img_coords_filtrados:
        transformado = transformar_punto(centro_img, matriz_perspectiva)
        if np.isinf(transformado[0]): continue

        es_unica_y_nueva = True
        for h_obj in historial_existente:
            if h_obj['clase'] == clase_actual:
                dist_espacial = np.linalg.norm(transformado - h_obj['centro_transformado'])
                dist_temporal = frame_actual - h_obj['frame']
                if not (dist_espacial >= umbral_espacio or dist_temporal >= umbral_tiempo):
                    es_unica_y_nueva = False
                    break

        if es_unica_y_nueva:
            tacha_id_counter += 1
            nuevas_registradas.append({
                'id' : tacha_id_counter,
                'centro_transformado': transformado,
                'centro_img': centro_img,
                'frame': frame_actual,
                'clase': clase_actual
            })
    return nuevas_registradas

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

zona_x_inicio = int(frame_width * 0.20)
zona_x_fin = int(frame_width * 0.50)
zona_y_inicio = int(frame_height * 0.55)
zona_y_fin = frame_height

pts_origen = np.float32([
    [zona_x_inicio, zona_y_inicio],
    [zona_x_fin, zona_y_inicio],
    [zona_x_inicio, zona_y_fin],
    [zona_x_fin, zona_y_fin]
])
ancho_transformado = 400
alto_transformado = 600

pts_destino = np.float32([
    [0, 0],
    [ancho_transformado, 0],
    [0, alto_transformado],
    [ancho_transformado, alto_transformado]
])
M = cv2.getPerspectiveTransform(pts_origen, pts_destino)
M_inv = cv2.getPerspectiveTransform(pts_destino, pts_origen)

frame_idx = 0
temporal_candidate_history = []

# --- Sección de la Red Neuronal (NN) ---
# Clase para el modelo de la red neuronal
class ModeloPatronTachas(torch.nn.Module):
    def __init__(self, longitud_secuencia_entrada, tamano_oculto=64): # longitud_secuencia_entrada no usada directamente en def. de capas
        super(ModeloPatronTachas, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=tamano_oculto, batch_first=True)
        self.lineal = torch.nn.Linear(tamano_oculto, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        salida = self.lineal(lstm_out[:, -1, :]) # Usa la salida del último paso de tiempo
        return salida

# Valores por defecto, serán sobrescritos si se cargan desde el checkpoint
RUTA_MODELO_NN = 'NNPT/modelo_patron_tachas.pt'
LONGITUD_SECUENCIA_ENTRADA_NN = 5
MEDIA_DIST_ENTRENAMIENTO = 50.0
STD_DIST_ENTRENAMIENTO = 10.0

# --- Instanciar modelo ---
modelo_patron_tachas = ModeloPatronTachas(LONGITUD_SECUENCIA_ENTRADA_NN).to(device)
NN_MODELO_CARGADO = False
nn_model_status_message = ""

# Carga del modelo y parámetros guardados
try:
    checkpoint = torch.load(RUTA_MODELO_NN, map_location=device)

    LONGITUD_SECUENCIA_ENTRADA_NN = checkpoint.get('longitud_secuencia', LONGITUD_SECUENCIA_ENTRADA_NN)
    # Si la arquitectura del modelo dependiera estrictamente de LONGITUD_SECUENCIA_ENTRADA_NN al instanciarse, se debería re-instanciar el modelo aquí con el valor cargado antes de cargar state_dict.
    # ej: modelo_patron_tachas = ModeloPatronTachas(LONGITUD_SECUENCIA_ENTRADA_NN).to(device)

    modelo_patron_tachas.load_state_dict(checkpoint['model_state_dict'])
    MEDIA_DIST_ENTRENAMIENTO = checkpoint.get('media_entrenamiento', MEDIA_DIST_ENTRENAMIENTO)
    STD_DIST_ENTRENAMIENTO = checkpoint.get('std_entrenamiento', STD_DIST_ENTRENAMIENTO)

    modelo_patron_tachas.eval() # Poner el modelo en modo de evaluación
    NN_MODELO_CARGADO = True
    nn_model_status_message = f"Modelo NN y parámetros cargados exitosamente desde '{RUTA_MODELO_NN}'."
    print(nn_model_status_message)
    print(f"  Longitud de Secuencia: {LONGITUD_SECUENCIA_ENTRADA_NN}, Media Entrenamiento: {MEDIA_DIST_ENTRENAMIENTO:.2f}, Std Dev Entrenamiento: {STD_DIST_ENTRENAMIENTO:.2f}")

except FileNotFoundError:
    nn_model_status_message = f"Archivo del modelo NN '{RUTA_MODELO_NN}' no encontrado. Se utilizará un modelo nuevo y parámetros por defecto/calculados."
    print(nn_model_status_message)
    NN_MODELO_CARGADO = False
except Exception as e:
    nn_model_status_message = f"Error al cargar el modelo NN desde '{RUTA_MODELO_NN}': {e}. La validación NN podría ser omitida o usar heurísticas."
    print(nn_model_status_message)
    modelo_patron_tachas = None # Indicar que el modelo no es utilizable
    NN_MODELO_CARGADO = False

if not NN_MODELO_CARGADO and modelo_patron_tachas is not None:
    # Este mensaje se muestra si el archivo no se encontró y estamos usando un modelo nuevo que no se haya entrenado
    new_model_msg = (f"Usando modelo NN no entrenado con parámetros iniciales (serán actualizados si aplica más adelante):\n"
                     f"  Longitud Secuencia={LONGITUD_SECUENCIA_ENTRADA_NN}, Media={MEDIA_DIST_ENTRENAMIENTO:.2f}, StdDev={STD_DIST_ENTRENAMIENTO:.2f}")
    print(new_model_msg)
    if not nn_model_status_message: # Si no hubo un error previo, actualizar el mensaje
        nn_model_status_message = new_model_msg


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tiempo_actual = frame_idx / fps if fps > 0 else 0
    frame_corregido = cv2.undistort(frame, K, D)

    captafaros_raw, tachas_raw, senaleticas_raw = detectar_objetos(frame_corregido)

    # Futuros candidatos para filtrar tachas
    tachas_candidatas_info_actual = filtrar_detecciones(
        tachas_raw, captafaros_raw, senaleticas_raw,
        zona_x_inicio, zona_x_fin, zona_y_inicio, zona_y_fin,
        frame_idx
    )
    
    #Historial de candidatos temporales
    temporal_candidate_history = [
        cand for cand in temporal_candidate_history
        if frame_idx - cand['frame_idx'] < MAX_CANDIDATE_AGE_FRAMES
    ]

    # Tachas persistentes para registro
    confirmed_persistent_tachas_for_registration = []

    for current_cand in tachas_candidatas_info_actual:
        frames_found_in = {current_cand['frame_idx']}

        for hist_cand in temporal_candidate_history:
            if hist_cand['frame_idx'] == current_cand['frame_idx']:
                continue

            dist = np.linalg.norm([current_cand['cx'] - hist_cand['cx'], current_cand['cy'] - hist_cand['cy']])

            if dist < PERSISTENCE_PIXEL_TOLERANCE:
                frames_found_in.add(hist_cand['frame_idx'])

        if len(frames_found_in) >= MIN_HITS_FOR_PERSISTENCE:
            confirmed_persistent_tachas_for_registration.append(np.array([current_cand['cx'],current_cand['cy']]))
    
    # Antiguo filtrado de tacha
    # tachas_filtradas_img_coords = filtrar_detecciones(tachas_raw, captafaros_raw, senaleticas_raw,
    #                                             zona_x_inicio, zona_x_fin, zona_y_inicio, zona_y_fin)

    # nuevas_tachas_este_frame = registrar_detecciones_nuevas(
    #     tachas_filtradas_img_coords, 'tacha', objeto_historial, frame_idx, M
    # )

    # Si se comenta, deja la muestra de tachas en falsos positivos, en caso de que no, no muestra nada por pantalla (investigar)
    temporal_candidate_history.extend(tachas_candidatas_info_actual)

    nuevas_tachas_este_frame = registrar_detecciones_nuevas(
        confirmed_persistent_tachas_for_registration,
        'tacha',
        objeto_historial,
        frame_idx,
        M
    )

    objeto_historial.extend(nuevas_tachas_este_frame)

    desviacion_frames_calc = 0 # Inicializar
    if nuevas_tachas_este_frame:
        if len(detecciones_frames) == 0 or (frame_idx - detecciones_frames[-1]) >= min_frames_entre_detecciones:
            detecciones_frames.append(frame_idx)
            if len(detecciones_frames) >= ventana_inicial + 1:
                intervalos = [j - i for i, j in zip(detecciones_frames[:-1], detecciones_frames[1:])]
                promedio_frames_calc = sum(intervalos[-ventana_inicial:]) / ventana_inicial
                desviacion_frames_calc = statistics.stdev(intervalos[-ventana_inicial:]) if len(intervalos[-ventana_inicial:]) > 1 else 0
                if promedio_frames_calc > 0:
                    promedio_frames = promedio_frames_calc
                print(f"[DEBUG ALERTA TACHA] Frame: {frame_idx}, Nueva tacha detectada. Promedio Intervalo Frames: {(f'{promedio_frames:.2f}' if promedio_frames is not None else 'N/A')}, Desv: {desviacion_frames_calc:.2f}")


    if promedio_frames and len(detecciones_frames) > ventana_inicial:
        tiempo_sin_deteccion = frame_idx - detecciones_frames[-1]
        current_desviacion = desviacion_frames_calc if 'desviacion_frames_calc' in locals() and desviacion_frames_calc > 0 else (promedio_frames * 0.1 if promedio_frames else 0.1)
        if tiempo_sin_deteccion > promedio_frames + tolerancia_frames * current_desviacion:
            print(f"[ALERTA] Posible falta de tacha en frame {frame_idx}. Tiempo sin detección: {tiempo_sin_deteccion}")
            cv2.putText(frame_corregido, "ALERTA: FALTA TACHA?", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.rectangle(frame_corregido, (zona_x_inicio, zona_y_inicio), (zona_x_fin, zona_y_fin), (255, 255, 0), 2)

    tachas_a_dibujar = [h for h in objeto_historial if h['clase'] == 'tacha']
    for tacha_hist in tachas_a_dibujar[-historial_max_len:]:
        pt_img_original = transformar_punto(tacha_hist['centro_transformado'], M_inv)
        if not np.isinf(pt_img_original[0]):
             center_x_img = int(pt_img_original[0])
             center_y_img = int(pt_img_original[1])
             cv2.circle(frame_corregido, (int(pt_img_original[0]), int(pt_img_original[1])), 5, (0, 255, 255), -1)

             tacha_id_val = tacha_hist.get('id', 'N/A')
             cv2.putText(frame_corregido, f"ID:{tacha_id_val}",
                         (center_x_img + 7, center_y_img - 7),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         0.4,
                         (255, 255,255),
                         1)


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
            # print(f"Error en odometría visual (OpenCV): {e}") # Opcional: loggear error de CV
            pass


    prev_gray, prev_kp, prev_des = gray, kp, des

    if len(gps_tiempos) > 0 and len(gps_trayectoria) > 0:
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

    out.write(frame_corregido)
    cv2.imshow('Procesamiento en tiempo real', cv2.resize(frame_corregido, (0, 0), fx=0.5, fy=0.5)) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_idx % 100 == 0:
        print(f"Frame {frame_idx} procesado...")
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# --- ANÁLISIS DE PATRÓN DE COLOCACIÓN DE TACHAS ---
print("\n--- Iniciando Análisis de Patrón de Colocación de Tachas ---")

historial_solo_tachas = [obj for obj in objeto_historial if obj['clase'] == 'tacha']
total_detecciones_unicas_tachas_analisis = 0
info_analisis_tachas = [] # Para DataFrame Excel

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
    if total_detecciones_unicas_tachas_analisis <= 1: # Si hay 0 o 1 tacha
        info_analisis_tachas.append({"Métrica": "Nota sobre cálculo de distancias", "Valor": "No hay suficientes tachas (se necesitan al menos 2) para calcular distancias."})


media_espaciado_pix_bev = 0
std_dev_espaciado_pix_bev = 0
if distancias_entre_tachas:
    media_espaciado_pix_bev = np.mean(distancias_entre_tachas)
    std_dev_espaciado_pix_bev = np.std(distancias_entre_tachas)

    print("\nResultados del Análisis de Espaciado de Tachas (en píxeles de vista de pájaro):")
    print(f"  Número de distancias consecutivas calculadas: {len(distancias_entre_tachas)}")
    print(f"  Media del espaciado: {media_espaciado_pix_bev:.2f} píxeles (vista de pájaro)")
    print(f"  Desviación estándar del espaciado: {std_dev_espaciado_pix_bev:.2f} píxeles (vista de pájaro)")

    info_analisis_tachas.append({"Métrica": "Número de distancias consecutivas calculadas", "Valor": len(distancias_entre_tachas)})
    info_analisis_tachas.append({"Métrica": "Media del espaciado (píxeles BEV)", "Valor": f"{media_espaciado_pix_bev:.2f}"})
    info_analisis_tachas.append({"Métrica": "Desviación estándar del espaciado (píxeles BEV)", "Valor": f"{std_dev_espaciado_pix_bev:.2f}"})
    # Guardar todas las distancias para el informe Excel
    for i, dist in enumerate(distancias_entre_tachas):
        info_analisis_tachas.append({"Métrica": f"Distancia Tacha {i}-{i+1} (BEV)", "Valor": f"{dist:.2f}"})


    if not NN_MODELO_CARGADO and modelo_patron_tachas is not None:
        MEDIA_DIST_ENTRENAMIENTO = media_espaciado_pix_bev if media_espaciado_pix_bev > 0 else MEDIA_DIST_ENTRENAMIENTO
        STD_DIST_ENTRENAMIENTO = std_dev_espaciado_pix_bev if std_dev_espaciado_pix_bev > 0 else STD_DIST_ENTRENAMIENTO
        msg_params_actualizados = f"(Parámetros NN actualizados con datos del video: Media={MEDIA_DIST_ENTRENAMIENTO:.2f}, StdDev={STD_DIST_ENTRENAMIENTO:.2f})"
        print(msg_params_actualizados)
        info_analisis_tachas.append({"Métrica": "Actualización de Parámetros NN (basado en video actual)", "Valor": msg_params_actualizados})


else:
    print("No se calcularon distancias entre tachas.")
    if total_detecciones_unicas_tachas_analisis > 1: # Si había más de 1 tacha pero aún así no se calcularon distancias (extraño)
        info_analisis_tachas.append({"Métrica": "Advertencia sobre cálculo de distancias", "Valor": "No se calcularon distancias entre tachas a pesar de haber suficientes."})
    elif not total_detecciones_unicas_tachas_analisis <=1 : # Solo si no se añadió antes
        info_analisis_tachas.append({"Métrica": "Nota sobre cálculo de distancias", "Valor": "No se calcularon distancias entre tachas."})


print("--- Fin del Análisis de Patrón de Colocación de Tachas ---")

# --- Preparación de datos para la hoja "Datos NN" ---
info_nn = []
info_nn.append({"Parámetro": "Estado del Modelo NN", "Valor": nn_model_status_message})
info_nn.append({"Parámetro": "Modelo NN Cargado", "Valor": NN_MODELO_CARGADO})
info_nn.append({"Parámetro": "Ruta Modelo NN", "Valor": RUTA_MODELO_NN})
info_nn.append({"Parámetro": "Longitud Secuencia Entrada NN (Actual)", "Valor": LONGITUD_SECUENCIA_ENTRADA_NN})
info_nn.append({"Parámetro": "Media Distancia Entrenamiento NN (Actual)", "Valor": f"{MEDIA_DIST_ENTRENAMIENTO:.2f}"})
info_nn.append({"Parámetro": "Std Dev Distancia Entrenamiento NN (Actual)", "Valor": f"{STD_DIST_ENTRENAMIENTO:.2f}"})

alertas_tacha_faltante_nn = 0
validation_nn_message = ""

if distancias_entre_tachas and len(distancias_entre_tachas) >= LONGITUD_SECUENCIA_ENTRADA_NN:
    if NN_MODELO_CARGADO and modelo_patron_tachas is not None:
        validation_nn_message = "--- Validando Patrón de Tacha con Red Neuronal Entrenada ---"
        print(f"\n{validation_nn_message}")
    elif modelo_patron_tachas is not None: # Modelo existe pero no fue cargado (es nuevo/no entrenado)
        validation_nn_message = "--- (Validación NN con Modelo No Entrenado o Heurísticas) ---\n--- (Verificación Ilustrativa estilo NN en Secuencias con parámetros actuales) ---"
        print(f"\n{validation_nn_message}")
    else: # No hay modelo
        validation_nn_message = "--- (Validación NN OMITIDA - Modelo No Disponible) ---"
        print(f"\n{validation_nn_message}")

    if modelo_patron_tachas is not None: # Solo proceder si hay un objeto modelo
        for i in range(len(distancias_entre_tachas) - LONGITUD_SECUENCIA_ENTRADA_NN + 1):
            secuencia_distancias = distancias_entre_tachas[i : i + LONGITUD_SECUENCIA_ENTRADA_NN]

            if NN_MODELO_CARGADO:
                try:
                    std_para_norm = STD_DIST_ENTRENAMIENTO if STD_DIST_ENTRENAMIENTO > 0 else 1.0
                    secuencia_normalizada = (np.array(secuencia_distancias) - MEDIA_DIST_ENTRENAMIENTO) / std_para_norm
                    tensor_entrada = torch.tensor(secuencia_normalizada, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)

                    with torch.no_grad():
                        prediccion = modelo_patron_tachas(tensor_entrada)
                        prob_faltante = torch.sigmoid(prediccion).item()

                        if prob_faltante > 0.75:
                            alertas_tacha_faltante_nn += 1
                            idx_ultima_tacha_en_secuencia = i + LONGITUD_SECUENCIA_ENTRADA_NN
                            msg_alerta_nn = ""
                            if idx_ultima_tacha_en_secuencia < len(historial_tachas_ordenado):
                                info_tacha_afectada = historial_tachas_ordenado[idx_ultima_tacha_en_secuencia]
                                msg_alerta_nn = (f"ALERTA NN (Modelo Entrenado): Patrón de tacha potencialmente faltante. "
                                      f"Prob: {prob_faltante:.2f}. Secuencia terminando cerca de la tacha en pos BEV {info_tacha_afectada['centro_transformado']} (frame {info_tacha_afectada['frame']}). "
                                      f"Distancias: {[round(s,1) for s in secuencia_distancias]}")
                            else:
                                 msg_alerta_nn = (f"ALERTA NN (Modelo Entrenado): Patrón de tacha potencialmente faltante. Prob: {prob_faltante:.2f}. "
                                       f"Distancias: {[round(s,1) for s in secuencia_distancias]}")
                            print(f"  {msg_alerta_nn}")
                            info_nn.append({"Parámetro": f"Alerta NN Entrenado {alertas_tacha_faltante_nn}", "Valor": msg_alerta_nn})
                except Exception as e:
                    err_msg = f"Error durante la predicción de la NN: {e}"
                    print(f"  {err_msg}")
                    info_nn.append({"Parámetro": "Error Predicción NN", "Valor": err_msg})

            elif not NN_MODELO_CARGADO and modelo_patron_tachas is not None:
                media_secuencia_dist = np.mean(secuencia_distancias)
                factor_umbral = 2.0
                ref_media = MEDIA_DIST_ENTRENAMIENTO
                ref_std = STD_DIST_ENTRENAMIENTO if STD_DIST_ENTRENAMIENTO > 0 else (ref_media * 0.1 if ref_media > 0 else 1.0)

                if any(d > ref_media + factor_umbral * ref_std for d in secuencia_distancias) or \
                   (len(secuencia_distancias) > 1 and np.std(secuencia_distancias) > ref_std * 1.5) :
                    alertas_tacha_faltante_nn += 1
                    idx_ultima_tacha_en_secuencia = i + LONGITUD_SECUENCIA_ENTRADA_NN
                    msg_alerta_heuristica = ""
                    if idx_ultima_tacha_en_secuencia < len(historial_tachas_ordenado):
                        info_tacha_afectada = historial_tachas_ordenado[idx_ultima_tacha_en_secuencia]
                        msg_alerta_heuristica = (f"(ALERTA heurística/modelo no entrenado): Patrón de tacha potencialmente faltante. "
                              f"Secuencia terminando cerca de la tacha en pos BEV {info_tacha_afectada['centro_transformado']} (frame {info_tacha_afectada['frame']}). "
                              f"Distancias: {[round(s,1) for s in secuencia_distancias]}")
                    else:
                         msg_alerta_heuristica = (f"(ALERTA heurística/modelo no entrenado): Patrón de tacha potencialmente faltante. Distancias: {[round(s,1) for s in secuencia_distancias]}")
                    print(f"  {msg_alerta_heuristica}")
                    info_nn.append({"Parámetro": f"Alerta NN Heurística {alertas_tacha_faltante_nn}", "Valor": msg_alerta_heuristica})


    summary_alerts_msg = ""
    if alertas_tacha_faltante_nn > 0:
        summary_alerts_msg = f"Validación NN/Heurística: {alertas_tacha_faltante_nn} alertas potenciales de tacha faltante basadas en patrones de secuencia."
        print(f"  {summary_alerts_msg}")
    elif modelo_patron_tachas is not None and len(distancias_entre_tachas) >= LONGITUD_SECUENCIA_ENTRADA_NN :
        summary_alerts_msg = "Validación NN/Heurística: No se detectaron anomalías significativas en el patrón."
        print(f"  {summary_alerts_msg}")
    info_nn.append({"Parámetro": "Resumen Alertas NN/Heurística", "Valor": summary_alerts_msg})


elif not distancias_entre_tachas:
    validation_nn_message = "No hay distancias de tachas disponibles para la validación NN/Heurística."
    print(f"\n{validation_nn_message}")
else: # distancias_entre_tachas existe pero es menor que LONGITUD_SECUENCIA_ENTRADA_NN
    validation_nn_message = f"No hay suficientes distancias de tachas ({len(distancias_entre_tachas)}) para la validación NN/Heurística con longitud de secuencia {LONGITUD_SECUENCIA_ENTRADA_NN}."
    print(f"\n{validation_nn_message}")

info_nn.append({"Parámetro": "Mensaje Validación NN", "Valor": validation_nn_message})
info_nn.append({"Parámetro": "Total Alertas NN/Heurística Generadas", "Valor": alertas_tacha_faltante_nn})


# --- GUARDAR EL MODELO NN (SI EXISTE Y ES VÁLIDO) Y SUS PARÁMETROS ---
nn_save_status = "Modelo NN no disponible o no se intentó guardar."
if modelo_patron_tachas is not None:
    try:
        checkpoint = {
            'model_state_dict': modelo_patron_tachas.state_dict(),
            'longitud_secuencia': LONGITUD_SECUENCIA_ENTRADA_NN,
            'media_entrenamiento': MEDIA_DIST_ENTRENAMIENTO,
            'std_entrenamiento': STD_DIST_ENTRENAMIENTO
        }
        torch.save(checkpoint, RUTA_MODELO_NN)
        nn_save_status = f"Modelo NN y parámetros guardados exitosamente en '{RUTA_MODELO_NN}'."
        print(f"\n{nn_save_status}")
    except Exception as e:
        nn_save_status = f"Error al guardar el modelo NN y parámetros en '{RUTA_MODELO_NN}': {e}"
        print(f"\n{nn_save_status}")
info_nn.append({"Parámetro": "Estado Guardado Modelo NN", "Valor": nn_save_status})


# --- GUARDAR ANÁLISIS EN EXCEL ---
fecha_actual_str = datetime.now().strftime("%m-%d-%y")
nombre_archivo_excel = f"{fecha_actual_str} - Análisis FP SSGL.xlsx"

df_analisis_tachas = pd.DataFrame(info_analisis_tachas)
df_datos_nn = pd.DataFrame(info_nn)

try:
    with pd.ExcelWriter(nombre_archivo_excel, engine='openpyxl') as writer:
        df_analisis_tachas.to_excel(writer, sheet_name='Análisis Patrón Tachas', index=False)
        df_datos_nn.to_excel(writer, sheet_name='Datos NN', index=False)
    print(f"\nInforme de análisis guardado en: '{nombre_archivo_excel}'")
except Exception as e:
    print(f"\nError al guardar el informe de análisis en Excel: {e}")


if posiciones_vehiculo and gps_posiciones_vehiculo:
    odo_np = np.array(posiciones_vehiculo)
    gps_np = np.array(gps_posiciones_vehiculo)

    plt.figure(figsize=(10, 8))
    plt.plot(odo_np[:, 0], odo_np[:, 1], 'g-', label='Odometría Visual (Escala Relativa)')
    if gps_np.ndim == 2 and gps_np.shape[1] == 2: # Asegurarse que gps_np tiene la forma correcta
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
        #plt.show() # Comentado para ejecución sin display
        print("Gráfico de trayectoria comparativa guardado.")
    except Exception as e:
        print(f"Error al guardar el gráfico de trayectoria: {e}")
else:
    print("No hay suficientes datos de odometría o GPS para graficar la trayectoria comparativa.")

print("\nProcesamiento finalizado.")