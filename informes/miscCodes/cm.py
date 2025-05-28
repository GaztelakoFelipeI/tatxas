import os
import shutil
import stat
import nbformat
import subprocess
from nbconvert.preprocessors import ExecutePreprocessor

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def incluirLibMeta():
    ruta_carpeta = 'py-gpmf-parser'
    ruta_notebook = 'lecturaMetadatos.ipynb'
    marcador = os.path.join(ruta_carpeta, '.created_by_script')

    # Si la carpeta existe Y fue creada por este script, eliminarla
    if os.path.exists(ruta_carpeta) and os.path.exists(marcador):
        print(f"[INFO] Carpeta '{ruta_carpeta}' creada por el script. Eliminando...")
        try:
            shutil.rmtree(ruta_carpeta, onerror=on_rm_error)
        except Exception as e:
            print(f"[ERROR] No se pudo eliminar la carpeta: {e}")
            return

    # Clonar el repositorio si no existe
    if not os.path.exists(ruta_carpeta):
        print(f"[INFO] Clonando repositorio...")
        try:
            subprocess.run(
                ["git", "clone", "--recursive", "https://github.com/urbste/py-gpmf-parser/"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error al clonar el repositorio: {e}")
            return

    # Instalar la librería clonada localmente
    print("[INFO] Instalando py-gpmf-parser localmente...")
    try:
        subprocess.run(["pip", "install", "."], check=True, cwd=ruta_carpeta)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error al instalar py-gpmf-parser desde el repositorio: {e}")
        return

    # Instalar otras dependencias necesarias
    print("[INFO] Instalando otros paquetes requeridos...")
    try:
        subprocess.run(
            [
                "pip", "install",
                "numpy", "matplotlib", "gpxpy", "geopandas", "descartes", "contextily",
                "pandas", "ffmpeg-python", "gpmf", "folium",
                "selenium", "pyppeteer", "ultralytics"
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error al instalar paquetes adicionales: {e}")
        return

    # Ejecutar el notebook
    print(f"[INFO] Ejecutando notebook '{ruta_notebook}'...")
    with open(ruta_notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        try:
            ep.preprocess(nb, {'metadata': {'path': '.'}})
            print("[INFO] Ejecución del notebook completada exitosamente.")

            # Si se recreó la carpeta, dejar el marcador
            if os.path.exists(ruta_carpeta):
                with open(marcador, 'w') as m:
                    m.write('Esta carpeta fue creada por el script.\n')

        except Exception as e:
            print(f"[ERROR] Error al ejecutar el notebook: {e}")

incluirLibMeta()
