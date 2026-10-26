@echo off
rem Start Mayuri silently in the background
echo Starting Mayuri AI assistant in the background...
cd /d "%~dp0\.."
wscript.exe "%~dp0run_background.vbs"
echo Mayuri has been launched in the background.
echo Check logs at logs/mayuri.log and access UI at http://localhost:8000/ui/
pause
