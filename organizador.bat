@echo off
setlocal

rem Define las rutas de las carpetas de destino
set "DEST_EXCEL=informes\historialExcelInformes"
set "DEST_VIDEOS=informes\videosProcesados"

rem Crear las carpetas de destino si no existen
if not exist "%DEST_EXCEL%" (
    mkdir "%DEST_EXCEL%"
    echo Carpeta "%DEST_EXCEL%" creada.
)
if not exist "%DEST_VIDEOS%" (
    mkdir "%DEST_VIDEOS%"
    echo Carpeta "%DEST_VIDEOS%" creada.
)

echo.
echo Moviendo archivos Excel...
for %%f in (*.xls *.xlsx) do (
    if exist "%%f" (
        echo Moviendo "%%f" a "%DEST_EXCEL%\"
        move "%%f" "%DEST_EXCEL%\"
    )
)

echo.
echo Moviendo archivos de video MP4...
for %%f in (*.mp4) do (
    if exist "%%f" (
        echo Moviendo "%%f" a "%DEST_VIDEOS%\"
        move "%%f" "%DEST_VIDEOS%\"
    )
)

echo.
echo Proceso completado.
pause