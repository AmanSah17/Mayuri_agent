@echo off
rem Run Mayuri Standalone Frontend Server
echo =======================================================================
echo   Starting Mayuri Standalone Frontend UI Server (Port 3000)
echo =======================================================================
cd /d "%~dp0\..\mayuri\ui"

rem Detect active Python
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else if exist "D:\CUDA_ENV\CUDA_ENV\Scripts\python.exe" (
    set "PYTHON_EXE=D:\CUDA_ENV\CUDA_ENV\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python.exe"
)

echo Opening browser at http://localhost:3000 ...
start http://localhost:3000
"%PYTHON_EXE%" -m http.server 3000
pause
