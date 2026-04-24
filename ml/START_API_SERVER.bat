@echo off
echo ============================================================
echo   DarkGuard ML - API Server Startup Script
echo ============================================================
echo.
echo Starting ML API Server on http://127.0.0.1:5000
echo Keep this window open while using the Chrome extension.
echo Press Ctrl+C to stop the server.
echo.
cd /d "%~dp0"
python api_server.py
pause
