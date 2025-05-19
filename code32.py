# Mejora de odometría visual con acumulación de rotación y translación coherente
# Se ajusta el código original con una estimación más estable
# Y SE AÑADE ANÁLISIS DE PATRÓN DE COLOCACIÓN DE TACHAS

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
# Para la lógica de alerta y visualización, se podría usar una copia truncada si es necesario.
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
        
        # Comprobar si está demasiado cerca de objetos de las otras listas
        # Esto es para evitar detectar una tacha si está "encima" de un captafaro o señalética en la imagen
        demasiado_cerca_otros = False
        if len(otros1) > 0:
            for x1o, y1o, x2o, y2o, *_ in otros1:
                cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2
                if np.linalg.norm([cx - cxo, cy - cyo]) < 30: # Umbral de proximidad a OTROS tipos
                    demasiado_cerca_otros = True
                    break
        if demasiado_cerca_otros: continue

        if len(otros2) > 0:
            for x1o, y1o, x2o, y2o, *_ in otros2:
                cxo, cyo = (x1o + x2o) / 2, (y1o + y2o) / 2
                if np.linalg.norm([cx - cxo, cy - cyo]) < 30: # Umbral de proximidad a OTROS tipos
                    demasiado_cerca_otros = True
                    break
        if demasiado_cerca_otros: continue
        
        # Si no está cerca de OTROS tipos, se añade. La unicidad dentro de su PROPIA clase se maneja después.
        filtrados.append(np.array([cx, cy]))
    return filtrados


# MODIFICADA para añadir la clase al diccionario y renombrar 'centro' a 'centro_transformado' por claridad
def registrar_detecciones_nuevas(centros_img_coords_filtrados, clase_actual, historial_existente, frame_actual, matriz_perspectiva):
    nuevas_registradas = []
    for centro_img in centros_img_coords_filtrados:
        transformado = transformar_punto(centro_img, matriz_perspectiva)
        if np.isinf(transformado[0]): continue # Ocurrió una división por cero

        es_unica_y_nueva = True
        # Comprobar unicidad contra objetos de la MISMA clase en el historial
        for h_obj in historial_existente:
            if h_obj['clase'] == clase_actual:
                dist_espacial = np.linalg.norm(transformado - h_obj['centro_transformado'])
                dist_temporal = frame_actual - h_obj['frame']
                if not (dist_espacial >= umbral_espacio or dist_temporal >= umbral_tiempo):
                    es_unica_y_nueva = False
                    break
        
        if es_unica_y_nueva:
            nuevas_registradas.append({
                'centro_transformado': transformado, # Coordenadas en vista de pájaro
                'centro_img': centro_img,       # Coordenadas originales en la imagen (opcional, para depuración)
                'frame': frame_actual,
                'clase': clase_actual
            })
    return nuevas_registradas

# Video
ruta_video = 'videos/3.2 - 01_04 Tramo B1-B2.MP4'
output_video = '3.2 - 01_04 Tramo B1-B2_resultadoTransformado_conAnalisisTachas.MP4'
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
ancho_transformado = 400 # píxeles en la vista de pájaro
alto_transformado = 600  # píxeles en la vista de pájaro
# Es importante saber a qué distancia real (metros) corresponden estos píxeles
# para que el análisis de espaciado sea en unidades métricas.
# Por ahora, será en "píxeles de vista de pájaro".

pts_destino = np.float32([
    [0, 0],
    [ancho_transformado, 0],
    [0, alto_transformado],
    [ancho_transformado, alto_transformado]
])
M = cv2.getPerspectiveTransform(pts_origen, pts_destino)
M_inv = cv2.getPerspectiveTransform(pts_destino, pts_origen) # Inversa para dibujar

frame_idx = 0
# trayectoria_img = np.ones((600, 600, 3), dtype=np.uint8) * 255 # No usado directamente, pero conceptualmente similar al minimapa
# escala = 10 # Para trayectoria_img
# offset = 300 # Para trayectoria_img

# Historial global para todas las clases (si se quisiera extender)
# Por ahora, `objeto_historial` se usa principalmente para tachas debido al flujo original
# pero la estructura de datos ahora soporta clases.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tiempo_actual = frame_idx / fps
    frame_corregido = cv2.undistort(frame, K, D)
    
    captafaros_raw, tachas_raw, senaleticas_raw = detectar_objetos(frame_corregido)
    
    # Filtrar detecciones de TACHAS (lista de coordenadas [cx, cy] en la imagen original)
    tachas_filtradas_img_coords = filtrar_detecciones(tachas_raw, captafaros_raw, senaleticas_raw,
                                                zona_x_inicio, zona_x_fin, zona_y_inicio, zona_y_fin)

    # Registrar TACHAS nuevas. `objeto_historial` contendrá el historial acumulado de tachas.
    nuevas_tachas_este_frame = registrar_detecciones_nuevas(
        tachas_filtradas_img_coords, 'tacha', objeto_historial, frame_idx, M
    )
    objeto_historial.extend(nuevas_tachas_este_frame)
    # No se trunca objeto_historial aquí para permitir el análisis completo al final.
    # Si el rendimiento o memoria fueran un problema durante la ejecución, se podría truncar una copia para visualización.


    # Lógica de alerta de falta de tacha (basada en las *nuevas* tachas detectadas en este frame)
    if nuevas_tachas_este_frame: # Si se registró al menos una tacha nueva en este frame
        if len(detecciones_frames) == 0 or (frame_idx - detecciones_frames[-1]) >= min_frames_entre_detecciones:
            detecciones_frames.append(frame_idx)
            if len(detecciones_frames) >= ventana_inicial + 1:
                intervalos = [j - i for i, j in zip(detecciones_frames[:-1], detecciones_frames[1:])]
                promedio_frames = sum(intervalos[-ventana_inicial:]) / ventana_inicial
                desviacion_frames = statistics.stdev(intervalos[-ventana_inicial:]) if len(intervalos[-ventana_inicial:]) > 1 else 0
                print(f"[DEBUG TACHA ALERT] Frame: {frame_idx}, Nueva tacha detectada. Promedio Intervalo Frames: {promedio_frames:.2f}, Desv: {desviacion_frames:.2f}")

    if promedio_frames and len(detecciones_frames) > ventana_inicial: # Si ya tenemos una estimación del promedio
        tiempo_sin_deteccion = frame_idx - detecciones_frames[-1] # Tiempo desde la última NUEVA tacha
        if tiempo_sin_deteccion > promedio_frames + tolerancia_frames * desviacion_frames:
            print(f"[ALERTA] Posible falta de tacha en frame {frame_idx}. Tiempo sin detección: {tiempo_sin_deteccion}")
            cv2.putText(frame_corregido, "ALERTA: FALTA TACHA?", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Dibujar zona de interés y detecciones (solo las últimas N tachas para no saturar)
    cv2.rectangle(frame_corregido, (zona_x_inicio, zona_y_inicio), (zona_x_fin, zona_y_fin), (255, 255, 0), 2)
    
    # Mostrar solo las últimas 'historial_max_len' tachas detectadas para la visualización
    tachas_a_dibujar = [h for h in objeto_historial if h['clase'] == 'tacha']
    for tacha_hist in tachas_a_dibujar[-historial_max_len:]:
        # Para dibujar, necesitamos las coordenadas en la imagen original, no las transformadas.
        # `transformar_punto` ahora puede usar M_inv (perspectiva inversa)
        # El punto en `tacha_hist['centro_transformado']` está en la vista de pájaro.
        pt_img_original = transformar_punto(tacha_hist['centro_transformado'], M_inv)
        if not np.isinf(pt_img_original[0]):
             cv2.circle(frame_corregido, (int(pt_img_original[0]), int(pt_img_original[1])), 5, (0, 255, 255), -1)


    # Odometría visual
    gray = cv2.cvtColor(frame_corregido, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if prev_gray is not None and prev_des is not None and des is not None and len(kp) > 0:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 20:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Usar K para findEssentialMat y recoverPose
            E, mask_e = cv2.findEssentialMat(dst_pts, src_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, mask_rp = cv2.recoverPose(E, dst_pts, src_pts, K)
                
                # Escala: Este es el mayor desafío en VO monocular.
                # Aquí se asume una escala, pero en la realidad necesitarías otra fuente (GPS, IMU, altura conocida)
                # o triangulación de puntos 3D si conoces la distancia entre ellos.
                escala_asumida = 1.0 # Esto necesita calibración o estimación dinámica.
                                     # Por ahora, la trayectoria de VO es relativa en escala.
                
                t_f += escala_asumida * (R_f @ t)
                R_f = R @ R_f
                posiciones_vehiculo.append(np.array([t_f[0][0], t_f[2][0]])) # Usamos X y Z para el plano del suelo

    prev_gray, prev_kp, prev_des = gray, kp, des

    # Sincronización con GPS (aproximada)
    if len(gps_tiempos) > 0:
        idx_gps = (np.abs(gps_tiempos - tiempo_actual)).argmin()
        gps_posiciones_vehiculo.append(gps_trayectoria[idx_gps])

    # Mini-mapa incrustado en el frame (Odometría Visual)
    mini_mapa_h, mini_mapa_w = 200, 200 # Tamaño más pequeño
    mini_mapa = np.ones((mini_mapa_h, mini_mapa_w, 3), dtype=np.uint8) * 255
    escala_mapa_vo = 5 # Ajustar escala para que la trayectoria VO quepa
    centro_mapa_x, centro_mapa_y = mini_mapa_w // 2, mini_mapa_h // 2
    
    if len(posiciones_vehiculo) > 1:
        for i in range(1, len(posiciones_vehiculo)):
            # Usar las posiciones X,Z de la VO. Y es usualmente la altura.
            x1_vo, z1_vo = posiciones_vehiculo[i-1] 
            x2_vo, z2_vo = posiciones_vehiculo[i]
            # Dibujar: (z es 'adelante', x es 'lateral')
            p1_mapa = (int(x1_vo * escala_mapa_vo + centro_mapa_x), int(z1_vo * escala_mapa_vo + centro_mapa_y))
            p2_mapa = (int(x2_vo * escala_mapa_vo + centro_mapa_x), int(z2_vo * escala_mapa_vo + centro_mapa_y))
            # Asegurarse que los puntos están dentro de los límites del minimapa
            if (0 <= p1_mapa[0] < mini_mapa_w and 0 <= p1_mapa[1] < mini_mapa_h and
                0 <= p2_mapa[0] < mini_mapa_w and 0 <= p2_mapa[1] < mini_mapa_h):
                cv2.line(mini_mapa, p1_mapa, p2_mapa, (0, 0, 255), 1) # VO en Rojo

    # Incrustar minimapa
    if frame_corregido.shape[0] >= mini_mapa_h + 10 and frame_corregido.shape[1] >= mini_mapa_w + 10:
        frame_corregido[10:mini_mapa_h+10, frame_corregido.shape[1]-mini_mapa_w-10:frame_corregido.shape[1]-10] = mini_mapa


    out.write(frame_corregido)
    cv2.imshow('Procesamiento en tiempo real', cv2.resize(frame_corregido, (0, 0), fx=0.5, fy=0.5))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_idx % 100 == 0: # Imprimir menos frecuentemente
        print(f"Frame {frame_idx} procesado...")
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# --- ANÁLISIS DE PATRÓN DE COLOCACIÓN DE TACHAS ---
print("\n--- Iniciando Análisis de Patrón de Colocación de Tachas ---")

# 1. Historial lineal con detecciones por clases (ya tenemos `objeto_historial` para tachas)
#    Cada elemento es {'centro_transformado': np.array([x,y]), 'frame': int, 'clase': 'tacha'}
#    `centro_transformado` está en coordenadas de la vista de pájaro.

# Filtrar solo las tachas (aunque objeto_historial ya debería contener solo tachas según el flujo)
historial_solo_tachas = [obj for obj in objeto_historial if obj['clase'] == 'tacha']

# Ordenar las tachas detectadas. Asumimos que la carretera avanza principalmente a lo largo del eje Y
# en la vista de pájaro (alto_transformado > ancho_transformado).
# Si la carretera es más horizontal en la vista de pájaro, ordenar por 'x' (índice 0).
# También se podría ordenar por frame, pero el orden espacial es más directo para el espaciado.
if historial_solo_tachas:
    # Ordenar por la coordenada 'y' de la vista de pájaro (principalmente), luego por 'x' (secundariamente)
    historial_tachas_ordenado = sorted(historial_solo_tachas, key=lambda t: (t['centro_transformado'][1], t['centro_transformado'][0]))
    print(f"Total de detecciones de tachas únicas procesadas para análisis: {len(historial_tachas_ordenado)}")
else:
    historial_tachas_ordenado = []
    print("No se encontraron tachas en el historial para analizar.")

# 2. Calcular distancia entre detecciones consecutivas del mismo tipo.
distancias_entre_tachas = []
if len(historial_tachas_ordenado) > 1:
    for i in range(1, len(historial_tachas_ordenado)):
        tacha_anterior = historial_tachas_ordenado[i-1]
        tacha_actual = historial_tachas_ordenado[i]
        
        # Distancia euclidiana en el espacio de la vista de pájaro
        distancia = np.linalg.norm(tacha_actual['centro_transformado'] - tacha_anterior['centro_transformado'])
        distancias_entre_tachas.append(distancia)
        
        # Opcional: imprimir cada distancia calculada
        # print(f"Distancia entre tacha (frame {tacha_anterior['frame']}, pos {tacha_anterior['centro_transformado']}) y "
        #       f"tacha (frame {tacha_actual['frame']}, pos {tacha_actual['centro_transformado']}): {distancia:.2f} pix_bev")
else:
    print("No hay suficientes tachas (se necesitan al menos 2) para calcular distancias.")

# 3. Estimar la media y desviación estándar del espaciado.
if distancias_entre_tachas:
    media_espaciado_pix_bev = np.mean(distancias_entre_tachas)
    std_dev_espaciado_pix_bev = np.std(distancias_entre_tachas)
    
    print("\nResultados del Análisis de Espaciado de Tachas (en píxeles de vista de pájaro):")
    print(f"  Número de distancias consecutivas calculadas: {len(distancias_entre_tachas)}")
    print(f"  Media del espaciado: {media_espaciado_pix_bev:.2f} píxeles (vista de pájaro)")
    print(f"  Desviación estándar del espaciado: {std_dev_espaciado_pix_bev:.2f} píxeles (vista de pájaro)")
    # print(f"  Todas las distancias (pix_bev): {[round(d, 2) for d in distancias_entre_tachas]}")

    # Para convertir a metros, necesitarías un factor de escala: metros_por_pixel_bev
    # Ejemplo: si sabes que `alto_transformado` (600px) corresponde a, digamos, 30 metros en la carretera,
    # entonces metros_por_pixel_bev_y = 30.0 / alto_transformado.
    # La distancia euclidiana en píxeles necesitaría una conversión más cuidadosa si X e Y tienen escalas diferentes.
    # Si la escala es uniforme: metros_por_pixel_bev = (metros_reales_conocidos / pixeles_bev_correspondientes)
    # media_espaciado_metros = media_espaciado_pix_bev * metros_por_pixel_bev
    # std_dev_espaciado_metros = std_dev_espaciado_pix_bev * metros_por_pixel_bev
    # print(f"  (Estimación) Media del espaciado: {media_espaciado_metros:.2f} m (asumiendo factor de escala)")
    # print(f"  (Estimación) Desviación estándar del espaciado: {std_dev_espaciado_metros:.2f} m (asumiendo factor de escala)")

else:
    print("No se calcularon distancias entre tachas.")

print("--- Fin del Análisis de Patrón de Colocación de Tachas ---")


# Mostrar trayectoria comparativa (VO vs GPS)
if posiciones_vehiculo and gps_posiciones_vehiculo:
    odo_np = np.array(posiciones_vehiculo)
    gps_np = np.array(gps_posiciones_vehiculo)

    plt.figure(figsize=(10, 8))
    # Trayectoria Odometría Visual (X,Z)
    plt.plot(odo_np[:, 0], odo_np[:, 1], 'g-', label='Odometría Visual (Escala Relativa)')
    
    # Trayectoria GPS (Longitud, Latitud) - Normalizar para comparación visual si es necesario
    # Para una comparación más directa, ambas trayectorias deberían estar en el mismo sistema de coordenadas
    # y con la misma escala. La VO es relativa y el GPS es absoluto.
    # Aquí se grafican tal cual, pero sus escalas y orígenes son diferentes.
    # Una forma simple de alinear es centrar ambas en (0,0) para ver la forma.
    gps_norm_lon = gps_np[:, 1] - gps_np[0, 1] 
    gps_norm_lat = gps_np[:, 0] - gps_np[0, 0]
    # Podrías necesitar un factor de escala para que las magnitudes sean comparables
    # Esto es solo para una visualización muy básica de la forma.
    # plt.plot(gps_norm_lon * escala_apropiada, gps_norm_lat * escala_apropiada, 'b--', label='GPS (Normalizado, Forma)')
    plt.plot(gps_np[:, 1], gps_np[:, 0], 'b--', label='GPS (Longitud, Latitud)') # GPS original

    plt.title('Comparación de Trayectorias (VO y GPS)')
    plt.xlabel('X / Longitud')
    plt.ylabel('Z / Latitud')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Intenta mantener la misma escala en ambos ejes
    plt.savefig('trayectoria_comparativa_vo_gps.png', dpi=300)
    plt.show()
else:
    print("No hay suficientes datos de odometría o GPS para graficar la trayectoria comparativa.")


print("Procesamiento finalizado.")