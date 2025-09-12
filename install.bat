@echo off
echo Face Recognition System - Windows Installation
echo =============================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Starting installation...
echo.

REM Run the Python installation script
python install.py

if errorlevel 1 (
    echo.
    echo Installation encountered errors. You can try manual installation:
    echo   pip install -r requirements_core.txt
    echo   pip install -r requirements_advanced.txt
    echo.
    pause
    exit /b 1
)

echo.
echo Installation complete! You can now run the application:
echo   python app.py
echo.
pause
