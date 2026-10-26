@echo off
rem Run Mayuri Backend Server (FastAPI REST, WebSocket, LangGraph Brain, Memory)
echo =======================================================================
echo   Starting Mayuri Backend Server (API + WebSocket + AI Agent)
echo =======================================================================
cd /d "%~dp0\.."

rem Detect active Python
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else if exist "D:\CUDA_ENV\CUDA_ENV\Scripts\python.exe" (
    set "PYTHON_EXE=D:\CUDA_ENV\CUDA_ENV\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python.exe"
)

"%PYTHON_EXE%" -m mayuri.main --backend
pause
