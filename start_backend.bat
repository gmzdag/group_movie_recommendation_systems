@echo off
REM ========================================
REM Cinefuse Backend Starter Script
REM ========================================

echo.
echo ========================================
echo   Starting Cinefuse Backend Server
echo ========================================
echo.

REM Check if we're in the correct directory
if not exist "src\main.py" (
    echo [ERROR] main.py not found!
    echo Please run this script from the group_movie_recommendation_systems directory
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [INFO] Python found: 
python --version
echo.

REM Check if uvicorn is installed
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] uvicorn not found. Installing...
    pip install uvicorn[standard] fastapi
    echo.
)

REM Start the server
echo [INFO] Starting FastAPI server on http://127.0.0.1:8000
echo [INFO] API Documentation: http://127.0.0.1:8000/docs
echo [INFO] Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

REM Run uvicorn
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000

REM If server stops
echo.
echo ========================================
echo   Server stopped
echo ========================================
pause
