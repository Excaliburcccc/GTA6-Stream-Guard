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
if not exist ".venv\Scripts\python.exe" goto missing
if "%~1"=="" goto camera
".venv\Scripts\python.exe" app.py --file "%~f1" --blood-mode selective
goto done
:camera
".venv\Scripts\python.exe" app.py --camera 0 --blood-mode selective
:done
if errorlevel 1 pause
exit /b
:missing
echo Run setup.cmd first.
pause
exit /b 1
