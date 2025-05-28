# Mejora de odometría visual con acumulación de rotación y translación coherente
# Se ajusta el código original con una estimación más estable
import os
import cv2
import numpy as np
import torch
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import folium
import time
import webbrowser
from PIL import Image
from selenium import webdriver
from ultralytics import YOLO

#integrar Librerias gopro para agregar metadata del vídeo (A FUTURO)
#incluirLibMeta()

# Configuración
gpu_id = 1
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print("Usando dispositivo:", device)

# Modelos YOLO personalizados
model_captafaros = YOLO('captaPT/best.pt').to(device)
model_tachas = YOLO('tachasPT/best.pt').to(device)
model_senaleticas = YOLO('PkPT/best.pt').to(device)

# Parámetros
conf_threshold = 0.3
historial_max_len = 25
tolerancia_frames = 1.5
ventana_inicial = 5
min_frames_entre_detecciones = 40
umbral_espacio = 25
umbral_tiempo = 30

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

# Historial detecciones
detecciones_frames = []
promedio_frames = None
objeto_historial = []

# Trayectoria GPS
gps_data = pd.read_excel("metadata/3.2 - 01_04 Tramo B1-B2.xlsx")
gps_trayectoria = gps_data[['Latitud', 'Longitud']].values
gps_tiempos = gps_data['Tiempo'].values
gps_posiciones_vehiculo = []


# #Generación de mapa GPS (intento número 1)
# def generar_mapa_gps(gps_coords, filename = 'mapa_satelital.png'):
#     lat0, lon0 = gps_coords[0]
#     m = folium.Map(location=[lat0,lon0],zoom_start = 17, tiles = 'OpenStreetMap')

#     folium.PolyLine(gps_coords, color='blue', weight = 4.5, opacity = 0.8)
#     folium.Marker([lat0,lon0], tooltip = 'Inicio').add_to(m)

#     m.save('mapa_temp.html')

#     #SS
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--window-size=600,600')
#     driver = webdriver.Chrome(options=options)
#     driver.get('file://' + os.path.abspath('mapa_temp.html'))
#     time.sleep(2)
#     driver.save_screenshot(filename)
#     driver.quit()

#     #Quitar bordes
#     img = Image.open(filename)
#     img = img.crop((0, 0, 500, 500))
#     img.save(filename)
#     print("Mapa generado: ",filename)

# if not os.path.exists('mapa_satelital.png'):
#     generar_mapa_gps(gps_trayectoria)

# #Generación de mapa GPS (intento número 3)
def generar_mapa_ruta_dinamico(gps_coords_recorridos, punto_actual, idx_frame):
    lat0, lon0 = gps_coords_recorridos[0]
    m = folium.Map(location=[lat0,lon0], zoom_start=17, tiles = 'OpenStreetMap')

    folium.PolyLine(gps_coords_recorridos, color='blue', weight = 4.5, opacity = 0.8).add_to(m)

    folium.Marker(location=punto_actual, tooltip = 'Posición actual',
                  icon = folium.Icon(color = 'red', icon='car', prefix='fa')).add_to(m)
    
    temp_html = f'mapas_dinamicos/mapa_frame_{idx_frame:05d}.html'
    os.makedirs('mapas_dinamicos', exist_ok = True)
    m.save(temp_html)

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--window-size=600x600')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)
    driver.get('file://' + os.path.abspath(temp_html))
    time.sleep(10)
    screenshot_path = f'mapas_dinamicos/mapa_frame{idx_frame:05d}.png'
    driver.save_screenshot(screenshot_path)
    driver.quit()

    img = Image.open(screenshot_path)
    img_array = np.array(img)

    non_blacks_rows = np.where(img_array.mean(axis=(1, 2)) > 5)[0]
    if len(non_blacks_rows) > 0:
        last_valid_row = non_blacks_rows[-1]
        img_cropped = img.crop((0, 0, img.width, last_valid_row + 1))
        img_cropped.save(screenshot_path)
        print(f"[MAPA] Mapa dinámico generado en {screenshot_path}")

# Funciones auxiliares
def transformar_punto(punto, matriz):
    punto_homog = np.array([[punto[0], punto[1], 1.0]]).T
    transformado = np.dot(matriz, punto_homog)
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
        x1b, y1b, x2b, y2b, conf, cls = obj
        if conf < conf_threshold:
            continue
        cx, cy = (x1b + x2b) / 2, (y1b + y2b) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        cerca = any(np.linalg.norm([(x1b + x2b) / 2 - (x1o + x2o) / 2, (y1b + y2b) / 2 - (y1o + y2o) / 2]) < 30
                    for x1o, y1o, x2o, y2o, *_ in np.vstack((otros1, otros2)))
        if not cerca:
            filtrados.append(np.array([cx, cy]))
    return filtrados

def registrar_detecciones_nuevas(filtrados, historial, frame_actual, matriz_perspectiva):
    nuevas = []
    for centro in filtrados:
        transformado = transformar_punto(centro, matriz_perspectiva)
        if all(np.linalg.norm(transformado - h['centro']) >= umbral_espacio or frame_actual - h['frame'] >= umbral_tiempo
               for h in historial):
            nuevas.append({'centro': transformado, 'frame': frame_actual})
    return nuevas

# Video
ruta_video = 'videos/3.2 - 01_04 Tramo B1-B2.MP4'
output_video = '3.2 - 01_04 Tramo B1-B2_resultadoTransformado.MP4'
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

frame_idx = 0
trayectoria_img = np.ones((600, 600, 3), dtype=np.uint8) * 255
escala = 10
offset = 300

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tiempo_actual = frame_idx / fps
    frame = cv2.undistort(frame, K, D)
    captafaros, tachas, senaleticas = detectar_objetos(frame)
    objetos_filtrados = filtrar_detecciones(tachas, captafaros, senaleticas,
                                            zona_x_inicio, zona_x_fin, zona_y_inicio, zona_y_fin)

    nuevas = registrar_detecciones_nuevas(objetos_filtrados, objeto_historial, frame_idx, M)
    objeto_historial.extend(nuevas)
    objeto_historial = objeto_historial[-historial_max_len:]
    objetos_unicos = [h['centro'] for h in objeto_historial]

    if nuevas:
        if len(detecciones_frames) == 0 or (frame_idx - detecciones_frames[-1]) >= min_frames_entre_detecciones:
            detecciones_frames.append(frame_idx)
            if len(detecciones_frames) >= ventana_inicial + 1:
                intervalos = [j - i for i, j in zip(detecciones_frames[:-1], detecciones_frames[1:])]
                promedio_frames = sum(intervalos[-ventana_inicial:]) / ventana_inicial
                desviacion_frames = statistics.stdev(intervalos[-ventana_inicial:]) if len(intervalos) > 1 else 0
                print(f"[DEBUG] Promedio: {promedio_frames:.2f}, desv: {desviacion_frames:.2f}")

    if promedio_frames and len(detecciones_frames) > ventana_inicial:
        tiempo = frame_idx - detecciones_frames[-1]
        if tiempo > promedio_frames + tolerancia_frames * desviacion_frames:
            print(f"[ALERTA] Posible falta de tacha en frame {frame_idx}")
            cv2.putText(frame, "FALTA TACHA", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.rectangle(frame, (zona_x_inicio, zona_y_inicio), (zona_x_fin, zona_y_fin), (255, 255, 0), 2)
    for centro in objetos_unicos:
        pt = transformar_punto(centro, np.linalg.inv(M))
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if prev_gray is not None and prev_des is not None and des is not None:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 20:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])
            E, _ = cv2.findEssentialMat(dst_pts, src_pts, focal=1.0, pp=(0., 0.), method=cv2.RANSAC)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, dst_pts, src_pts)
                escala_asumida = 1.0
                t_f += escala_asumida * R_f @ t
                R_f = R @ R_f
                posiciones_vehiculo.append(np.array([t_f[0][0], t_f[2][0]]))

    prev_gray, prev_kp, prev_des = gray, kp, des

    idx_gps = (np.abs(gps_tiempos - tiempo_actual)).argmin()
    gps_posiciones_vehiculo.append(gps_trayectoria[idx_gps])

    # Mini-mapa incrustado en el frame
    #mini_mapa = np.ones((500, 500, 3), dtype=np.uint8) * 255
    escala_mapa = 20
    mini_mapa = cv2.imread('screenshots/mapa_satelital.png')
    if mini_mapa is not None:
        mini_mapa = cv2.resize(mini_mapa, (500, 500))
        if len(mini_mapa.shape) == 2:
            mini_mapa = cv2.cvtColor(mini_mapa, cv2.COLOR_GRAY2BGR)
        for i in range(1, len(posiciones_vehiculo)):
            x1, y1 = posiciones_vehiculo[i - 1]
            x2, y2 = posiciones_vehiculo[i]
            p1 = (int(x1 * escala_mapa + 250), int(y1 * escala_mapa + 250))
            p2 = (int(x2 * escala_mapa + 250), int(y2 * escala_mapa + 250))
            cv2.line(mini_mapa, p1, p2, (0, 0, 255), 1)
        if frame.shape[0] >= 510 and frame.shape[1] >= 510:
            frame[10:510, -510:-10] = mini_mapa

        out.write(frame)
        cv2.imshow('Procesamiento en tiempo real', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_idx % 100 == 0 and len(gps_posiciones_vehiculo) > 1:
            gps_coords_hasta_ahora = gps_posiciones_vehiculo.copy()
            punto_actual = gps_coords_hasta_ahora[-1]
            print(f"[DEBUG] Generando mapa para frame {frame_idx}, GPS actual: {punto_actual}")
            generar_mapa_ruta_dinamico(gps_coords_hasta_ahora, punto_actual, frame_idx)
            print("Primeros puntos GPS:", gps_coords_hasta_ahora[:5])

        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx} procesado...")
        frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Mostrar trayectoria comparativa
odo = np.array(posiciones_vehiculo)
gps = np.array(gps_posiciones_vehiculo)

plt.plot(odo[:, 0], odo[:, 1], 'g-', label='Odometría')
plt.plot(gps[:, 1], gps[:, 0], 'b--', label='GPS')
plt.title('Comparación de Trayectorias')
plt.xlabel('X / Longitud')
plt.ylabel('Z / Latitud')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('screeshots/trayectoria_comparativa.png', dpi=300)
plt.show()

print("Procesamiento finalizado.")
