from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
import pandas as pd
import numpy as np
import os

def crear_dataframe(lat_lon_array, tiempo_array):
    # Asegurarse de que el arreglo tiene al menos dos columnas para latitud y longitud
    if lat_lon_array.shape[1] < 2:
        raise ValueError("El arreglo de coordenadas debe tener al menos dos columnas (latitud y longitud).")

    # Asegurarse de que los tamaños de los arreglos coincidan
    if len(lat_lon_array) != len(tiempo_array):
        raise ValueError("Los arreglos de coordenadas y tiempo deben tener la misma longitud.")

    # Extraer latitud y longitud
    latitudes = lat_lon_array[:, 0]
    longitudes = lat_lon_array[:, 1]

    # Crear el DataFrame
    df = pd.DataFrame({
        'Tiempo': tiempo_array,
        'Latitud': latitudes,
        'Longitud': longitudes,
    })

    return df

drive_directory = 'videos/'

print(f"Contenido de {drive_directory}:")
try:
    for item in os.listdir(drive_directory):
        print(item)
except FileNotFoundError:
    print(f"Error: El directorio {drive_directory} no fue encontrado. Asegúrate de que Google Drive esté montado correctamente.")
except Exception as e:
    print(f"Ocurrió un error: {e}")

file_path = drive_directory

if os.path.exists(file_path):
    print(True)
else:
    print(False)

video_directory = drive_directory
print(video_directory)


for filename in os.listdir(video_directory):
    if filename.endswith(('.MP4', '.MOV', '.mp4', '.mov')):  # Check for video file extensions
        video_path = os.path.join(video_directory, filename)
        print(f"Sacando datos GPS de: {video_path}")

        try:
            extractor = GoProTelemetryExtractor(video_path)
            extractor.open_source()
            gps, gps_t = extractor.extract_data('GPS5')
            extractor.close_source()

            df_armar_mapa = crear_dataframe(gps, gps_t)
            print(f"Finalizado para {filename}")
            df_armar_mapa.to_excel(f"/metadata/{filename[:-4]}.xlsx", index=False)
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

