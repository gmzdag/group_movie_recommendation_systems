@echo off
REM ========================================
REM Cinefuse - Start Both Backend & Frontend
REM ========================================

echo.
echo ========================================
echo   Starting Cinefuse Application
echo ========================================
echo.

REM Start backend in a new window
echo [INFO] Starting Backend Server...
start "Cinefuse Backend" cmd /k "cd /d %~dp0 && start_backend.bat"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in a new window
echo [INFO] Starting Frontend Server...
start "Cinefuse Frontend" cmd /k "cd /d %~dp0frontend && start_frontend.bat"

echo.
echo ========================================
echo   Both servers are starting...
echo ========================================
echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo.
echo Close the terminal windows to stop the servers
echo.
pause
