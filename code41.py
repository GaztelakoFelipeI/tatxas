# Mejora de odometría visual con acumulación de rotación y translación coherente
# Se ajusta el código original con una estimación más estable
# Y SE AÑADE ANÁLISIS DE PATRÓN DE COLOCACIÓN DE TACHAS
# + INTEGRACIÓN CONCEPTUAL DE RED NEURONAL PARA VALIDACIÓN DE PATRONES

import cv2
import numpy as np
import torch
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

# Configuración
gpu_id = 0 # SE CAMBIA A 0 SI SOLO HAY UNA GPU O ES LA PRIMERA
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print("Usando dispositivo:", device)

# Modelos YOLO personalizados
model_captafaros = YOLO('captaPT/best.pt').to(device)
model_tachas = YOLO('tachasPT/best.pt').to(device)
model_senaleticas = YOLO('PkPT/best.pt').to(device)

# Parámetros
conf_threshold = 0.3
historial_max_len = 100 # Aumentado para un mejor análisis de patrones, considerar hacerlo ilimitado para análisis post-video
tolerancia_frames = 1.5
ventana_inicial = 5
min_frames_entre_detecciones = 40 # Frames a esperar para considerar una nueva "ráfaga" de detecciones para el cálculo de promedio de alerta
umbral_espacio = 25 # Umbral de distancia (en píxeles de vista de pájaro) para considerar una detección como nueva
umbral_tiempo = 30  # Umbral de frames para considerar una detección como nueva aunque esté cerca espacialmente

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
detecciones_frames = [] # Historial de frames donde se detectaron NUEVAS tachas (para alerta de falta de tacha)
promedio_frames = None # Promedio de frames entre detecciones de tachas (para alerta)
objeto_historial = [] # Historial de todas las tachas detectadas {'centro_transformado', 'frame', 'clase'}

# Trayectoria GPS
gps_data = pd.read_excel("metadata/3.2 - 01_04 Tramo B1-B2.xlsx")
gps_trayectoria = gps_data[['Latitud', 'Longitud']].values
gps_tiempos = gps_data['Tiempo'].values
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

def filtrar_detecciones(objetos, otros1, otros2, x0, x1, y0, y1):
    filtrados = []
    for obj in objetos:
        x1b, y1b, x2b, y2b, conf, cls_id = obj # cls_id es el ID de la clase del modelo YOLO
        if conf < conf_threshold:
            continue
        cx, cy = (x1b + x2b) / 2, (y1b + y2b) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
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
        
        filtrados.append(np.array([cx, cy]))
    return filtrados

def registrar_detecciones_nuevas(centros_img_coords_filtrados, clase_actual, historial_existente, frame_actual, matriz_perspectiva):
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
            nuevas_registradas.append({
                'centro_transformado': transformado, 
                'centro_img': centro_img,       
                'frame': frame_actual,
                'clase': clase_actual
            })
    return nuevas_registradas

# Video
ruta_video = 'videos/3.2 - 01_04 Tramo B1-B2.MP4'
output_video = '3.2 - 01_04 Tramo B1-B2_resultadoTransformado_conAnalisisTachas_NN.MP4' # Changed output name
cap = cv2.VideoCapture(ruta_video)

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

# --- Conceptual: Load Neural Network Model (do this once) ---
# This is where you would load your pre-trained model for stud pattern validation.
# Ensure 'stud_pattern_model.pth' exists or comment out these lines if not using a real model yet.
NN_MODEL_LOADED = False
model_stud_pattern = None
NN_INPUT_SEQUENCE_LENGTH = 5  # Default, adjust if model loaded
MEAN_DIST_TRAINING = 50.0     # Default, adjust if model loaded
STD_DIST_TRAINING = 10.0      # Default, adjust if model loaded

try:
    if not NN_MODEL_LOADED: 
        print("(NN Model Placeholder: Skipping actual model load. Using dummy NN parameters for script structure.)")
except Exception as e:
    print(f"Error loading NN model: {e}. NN validation will be skipped.")
if not NN_MODEL_LOADED:
    NN_INPUT_SEQUENCE_LENGTH = 5
    MEAN_DIST_TRAINING = 50.0 
    STD_DIST_TRAINING = 10.0  


print(f"NN Model Loaded: {NN_MODEL_LOADED}, Sequence Length: {NN_INPUT_SEQUENCE_LENGTH}")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tiempo_actual = frame_idx / fps
    frame_corregido = cv2.undistort(frame, K, D)
    
    captafaros_raw, tachas_raw, senaleticas_raw = detectar_objetos(frame_corregido)
    
    tachas_filtradas_img_coords = filtrar_detecciones(tachas_raw, captafaros_raw, senaleticas_raw,
                                                zona_x_inicio, zona_x_fin, zona_y_inicio, zona_y_fin)

    nuevas_tachas_este_frame = registrar_detecciones_nuevas(
        tachas_filtradas_img_coords, 'tacha', objeto_historial, frame_idx, M
    )
    objeto_historial.extend(nuevas_tachas_este_frame)

    if nuevas_tachas_este_frame: 
        if len(detecciones_frames) == 0 or (frame_idx - detecciones_frames[-1]) >= min_frames_entre_detecciones:
            detecciones_frames.append(frame_idx)
            if len(detecciones_frames) >= ventana_inicial + 1:
                intervalos = [j - i for i, j in zip(detecciones_frames[:-1], detecciones_frames[1:])]
                promedio_frames_calc = sum(intervalos[-ventana_inicial:]) / ventana_inicial
                desviacion_frames_calc = statistics.stdev(intervalos[-ventana_inicial:]) if len(intervalos[-ventana_inicial:]) > 1 else 0
                # Update global promedio_frames only if calculation is valid
                if promedio_frames_calc > 0:
                    promedio_frames = promedio_frames_calc
                print(f"[DEBUG TACHA ALERT] Frame: {frame_idx}, Nueva tacha detectada. Promedio Intervalo Frames: {(f'{promedio_frames:.2f}' if promedio_frames is not None else 'N/A')}, Desv: {desviacion_frames_calc:.2f}")


    if promedio_frames and len(detecciones_frames) > ventana_inicial: 
        tiempo_sin_deteccion = frame_idx - detecciones_frames[-1] 
        # Use last calculated deviation; if not available, use a fraction of promedio_frames as a fallback
        current_desviacion = desviacion_frames_calc if 'desviacion_frames_calc' in locals() and desviacion_frames_calc > 0 else promedio_frames * 0.1 
        if tiempo_sin_deteccion > promedio_frames + tolerancia_frames * current_desviacion:
            print(f"[ALERTA] Posible falta de tacha en frame {frame_idx}. Tiempo sin detección: {tiempo_sin_deteccion}")
            cv2.putText(frame_corregido, "ALERTA: FALTA TACHA?", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.rectangle(frame_corregido, (zona_x_inicio, zona_y_inicio), (zona_x_fin, zona_y_fin), (255, 255, 0), 2)
    
    tachas_a_dibujar = [h for h in objeto_historial if h['clase'] == 'tacha']
    for tacha_hist in tachas_a_dibujar[-historial_max_len:]:
        pt_img_original = transformar_punto(tacha_hist['centro_transformado'], M_inv)
        if not np.isinf(pt_img_original[0]):
             cv2.circle(frame_corregido, (int(pt_img_original[0]), int(pt_img_original[1])), 5, (0, 255, 255), -1)

    gray = cv2.cvtColor(frame_corregido, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if prev_gray is not None and prev_des is not None and des is not None and len(kp) > 0:
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

    prev_gray, prev_kp, prev_des = gray, kp, des

    if len(gps_tiempos) > 0:
        idx_gps = (np.abs(gps_tiempos - tiempo_actual)).argmin()
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

if historial_solo_tachas:
    historial_tachas_ordenado = sorted(historial_solo_tachas, key=lambda t: (t['centro_transformado'][1], t['centro_transformado'][0]))
    print(f"Total de detecciones de tachas únicas procesadas para análisis: {len(historial_tachas_ordenado)}")
else:
    historial_tachas_ordenado = []
    print("No se encontraron tachas en el historial para analizar.")

distancias_entre_tachas = []
if len(historial_tachas_ordenado) > 1:
    for i in range(1, len(historial_tachas_ordenado)):
        tacha_anterior = historial_tachas_ordenado[i-1]
        tacha_actual = historial_tachas_ordenado[i]
        distancia = np.linalg.norm(tacha_actual['centro_transformado'] - tacha_anterior['centro_transformado'])
        distancias_entre_tachas.append(distancia)
else:
    print("No hay suficientes tachas (se necesitan al menos 2) para calcular distancias.")

media_espaciado_pix_bev = 0
std_dev_espaciado_pix_bev = 0
if distancias_entre_tachas:
    media_espaciado_pix_bev = np.mean(distancias_entre_tachas)
    std_dev_espaciado_pix_bev = np.std(distancias_entre_tachas)
    
    print("\nResultados del Análisis de Espaciado de Tachas (en píxeles de vista de pájaro):")
    print(f"  Número de distancias consecutivas calculadas: {len(distancias_entre_tachas)}")
    print(f"  Media del espaciado: {media_espaciado_pix_bev:.2f} píxeles (vista de pájaro)")
    print(f"  Desviación estándar del espaciado: {std_dev_espaciado_pix_bev:.2f} píxeles (vista de pájaro)")
    
    if not NN_MODEL_LOADED:
        MEAN_DIST_TRAINING = media_espaciado_pix_bev if media_espaciado_pix_bev > 0 else MEAN_DIST_TRAINING
        STD_DIST_TRAINING = std_dev_espaciado_pix_bev if std_dev_espaciado_pix_bev > 0 else STD_DIST_TRAINING
        print(f"(NN Placeholder: Using calculated mean {MEAN_DIST_TRAINING:.2f} and std {STD_DIST_TRAINING:.2f} for illustrative NN check)")

else:
    print("No se calcularon distancias entre tachas.")

print("--- Fin del Análisis de Patrón de Colocación de Tachas ---")

if distancias_entre_tachas and len(distancias_entre_tachas) >= NN_INPUT_SEQUENCE_LENGTH:
    if NN_MODEL_LOADED:
        print("\n--- Validating Stud Pattern with Trained Neural Network ---")
    else:
        print("\n--- (NN Validation SKIPPED - Model Not Loaded) ---")
        print("--- (Illustrative NN-style Check on Sequences) ---")

    missing_stud_alerts_nn = 0
    
    for i in range(len(distancias_entre_tachas) - NN_INPUT_SEQUENCE_LENGTH + 1):
        sequence_distances = distancias_entre_tachas[i : i + NN_INPUT_SEQUENCE_LENGTH]
        
        if NN_MODEL_LOADED and model_stud_pattern is not None:
            try:
                sequence_normalized = (np.array(sequence_distances) - MEAN_DIST_TRAINING) / (STD_DIST_TRAINING if STD_DIST_TRAINING > 0 else 1.0)
                input_tensor = torch.tensor(sequence_normalized, dtype=torch.float32).unsqueeze(0).to(device) 
                
                with torch.no_grad():
                    prediction = model_stud_pattern(input_tensor)
                    
                    prob_missing = torch.sigmoid(prediction).item() 
                    
                    if prob_missing > 0.75: 
                        missing_stud_alerts_nn += 1
                        last_stud_in_sequence_idx = i + NN_INPUT_SEQUENCE_LENGTH
                        if last_stud_in_sequence_idx < len(historial_tachas_ordenado):
                            affected_stud_info = historial_tachas_ordenado[last_stud_in_sequence_idx]
                            print(f"  NN ALERTA (Trained Model): Potential missing stud pattern. "
                                  f"Prob: {prob_missing:.2f}. Sequence ending near stud at BEV pos {affected_stud_info['centro_transformado']} (frame {affected_stud_info['frame']}). "
                                  f"Distances: {[round(s,1) for s in sequence_distances]}")
                        else:
                             print(f"  NN ALERTA (Trained Model): Potential missing stud pattern. Prob: {prob_missing:.2f}. "
                                   f"Distances: {[round(s,1) for s in sequence_distances]}")
            except Exception as e:
                print(f"  Error during NN prediction: {e}")
            
                NN_MODEL_LOADED = False

        if not NN_MODEL_LOADED or model_stud_pattern is None:
            mean_sequence_dist = np.mean(sequence_distances)
            threshold_factor = 2.0 
            
            ref_mean = media_espaciado_pix_bev if media_espaciado_pix_bev > 0 else MEAN_DIST_TRAINING
            ref_std = std_dev_espaciado_pix_bev if std_dev_espaciado_pix_bev > 0 else STD_DIST_TRAINING
            if ref_std == 0: ref_std = ref_mean * 0.1 if ref_mean > 0 else 1.0 

            if any(d > ref_mean + threshold_factor * ref_std for d in sequence_distances) or \
               (len(sequence_distances) > 1 and np.std(sequence_distances) > ref_std * 1.5) :
                missing_stud_alerts_nn += 1
                last_stud_in_sequence_idx = i + NN_INPUT_SEQUENCE_LENGTH
                if last_stud_in_sequence_idx < len(historial_tachas_ordenado):
                    affected_stud_info = historial_tachas_ordenado[last_stud_in_sequence_idx]
                    print(f"  (Illustrative NN-style ALERTA): Potential missing stud pattern. "
                          f"Sequence ending near stud at BEV pos {affected_stud_info['centro_transformado']} (frame {affected_stud_info['frame']}). "
                          f"Distances: {[round(s,1) for s in sequence_distances]}")
                else:
                     print(f"  (Illustrative NN-style ALERTA): Potential missing stud pattern. Distances: {[round(s,1) for s in sequence_distances]}")

    if missing_stud_alerts_nn > 0:
        print(f"  NN Validation: {missing_stud_alerts_nn} potential missing stud alerts based on sequence patterns.")
    elif len(distancias_entre_tachas) >= NN_INPUT_SEQUENCE_LENGTH: 
        print("  NN Validation: No significant pattern anomalies detected by (illustrative or trained) NN.")

elif not distancias_entre_tachas:
    print("\nNo stud distances available for NN validation.")
else: 
    print(f"\nNot enough stud distances ({len(distancias_entre_tachas)}) for NN validation with sequence length {NN_INPUT_SEQUENCE_LENGTH}.")


if posiciones_vehiculo and gps_posiciones_vehiculo:
    odo_np = np.array(posiciones_vehiculo)
    gps_np = np.array(gps_posiciones_vehiculo)

    plt.figure(figsize=(10, 8))
    plt.plot(odo_np[:, 0], odo_np[:, 1], 'g-', label='Odometría Visual (Escala Relativa)')
    plt.plot(gps_np[:, 1], gps_np[:, 0], 'b--', label='GPS (Longitud, Latitud)') 

    plt.title('Comparación de Trayectorias (VO y GPS)')
    plt.xlabel('X / Longitud')
    plt.ylabel('Z / Latitud')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trayectoria_comparativa_vo_gps.png', dpi=300)
    plt.show()
else:
    print("No hay suficientes datos de odometría o GPS para graficar la trayectoria comparativa.")

print("\nProcesamiento finalizado.")