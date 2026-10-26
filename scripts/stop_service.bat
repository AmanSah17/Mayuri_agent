@echo off
rem Terminate running Mayuri instances
echo Stopping Mayuri background processes...
taskkill /F /FI "WINDOWTITLE eq Mayuri*" >nul 2>&1
wmic process where "commandline like '%%mayuri.main%%'" delete >nul 2>&1
echo Done. Mayuri background service stopped.
pause
