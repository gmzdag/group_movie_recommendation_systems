@echo off
REM ========================================
REM Cinefuse Frontend Starter Script
REM ========================================

echo.
echo ========================================
echo   Starting Cinefuse Frontend (Vite)
echo ========================================
echo.

REM Check if we're in the correct directory
if not exist "package.json" (
    echo [ERROR] package.json not found!
    echo Please run this script from the frontend directory
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [INFO] Node.js found: 
node --version
echo.

REM Check if node_modules exists
if not exist "node_modules\" (
    echo [WARNING] node_modules not found. Installing dependencies...
    echo.
    npm install
    echo.
)

REM Start the dev server
echo [INFO] Starting Vite dev server on http://localhost:5173
echo [INFO] Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

REM Run npm dev
npm run dev

REM If server stops
echo.
echo ========================================
echo   Server stopped
echo ========================================
pause
