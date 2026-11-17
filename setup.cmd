@echo off
setlocal
set "GTA6_FILES=E:\CodexFiles"
if not exist "E:\." set "GTA6_FILES=%~dp0.runtime"
set "TEMP=%GTA6_FILES%\.tmp\codex\gta6-stream-guard"
set "TMP=%TEMP%"
set "TMPDIR=%TEMP%"
set "PIP_CACHE_DIR=%GTA6_FILES%\.cache\gta6-stream-guard\pip"
set "HF_HOME=%GTA6_FILES%\.cache\gta6-stream-guard\huggingface"
set "TORCH_HOME=%GTA6_FILES%\.cache\gta6-stream-guard\torch"
set "PYTHONUNBUFFERED=1"
if not exist "%TEMP%\." mkdir "%TEMP%"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto gpu
where py.exe >nul 2>&1
if not errorlevel 1 (
    for %%V in (3.12 3.11 3.13) do (
        call :try_launcher %%V
        if not errorlevel 1 goto gpu
    )
)
where python.exe >nul 2>&1
if errorlevel 1 goto python_missing
python -c "import struct,sys,tkinter; assert (3,11) <= sys.version_info[:2] <= (3,13) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto python_missing
python -m venv ".venv"
if errorlevel 1 goto failed

:gpu
where nvidia-smi.exe >nul 2>&1
if errorlevel 1 goto dependencies
".venv\Scripts\python.exe" -c "import torch; assert torch.__version__.startswith('2.8.0+cu128') and torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 goto dependencies
echo Installing PyTorch with CUDA 12.8 for NVIDIA acceleration...
".venv\Scripts\python.exe" -m pip install "torch==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 echo CUDA package could not be installed. Continuing with the standard dependencies.

:dependencies
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" download_police_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" download_scene_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import tkinter,cv2,onnxruntime,torch; onnxruntime.disable_telemetry_events(); from nudenet import NudeDetector; NudeDetector(); print('Capture, UI and detection ready. PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
if errorlevel 1 goto failed
where ffmpeg.exe >nul 2>&1
if errorlevel 1 echo Add ffmpeg.exe to PATH for local-file audio before using start.cmd.
where ffprobe.exe >nul 2>&1
if errorlevel 1 echo Add ffprobe.exe to PATH for local-file audio before using start.cmd.
echo Ready. Run start.cmd for the capture card or drop a local video onto start.cmd.
pause
exit /b 0

:try_launcher
py -%~1 -c "import struct,tkinter; assert struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 exit /b 1
py -%~1 -m venv ".venv"
exit /b %errorlevel%

:python_missing
echo Install 64-bit Python 3.11, 3.12 or 3.13 for Windows with Tcl/Tk and the py launcher.
pause
exit /b 1

:failed
echo Setup did not finish. Review the error above. You can run setup.cmd again.
pause
exit /b 1
