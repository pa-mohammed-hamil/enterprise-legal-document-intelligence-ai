@echo off
echo ================================================
echo Starting Frontend Server
echo ================================================
echo.
echo Frontend will be available at: http://localhost:3000
echo Backend API should be running at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

python -m http.server 3000
