@echo off
echo ========================================
echo Face Recognition System - Deployment Setup
echo ========================================

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "data" mkdir data
if not exist "static\css" mkdir static\css
if not exist "static\js" mkdir static\js
if not exist "static\images" mkdir static\images

echo.
echo Setting up environment configuration...
if not exist ".env" (
    copy ".env.example" ".env"
    echo Please edit .env file with your production settings
) else (
    echo .env file already exists
)

echo.
echo Setting up data files...
if not exist "data\students.csv" (
    echo student_id,name,email,department,year,registration_date,photo_path > data\students.csv
    echo Created students.csv
)

if not exist "data\attendance.csv" (
    echo student_id,name,date,time,status,confidence,method,latitude,longitude,location,city,state,country > data\attendance.csv
    echo Created attendance.csv
)

if not exist "data\training_metadata.json" (
    echo {} > data\training_metadata.json
    echo Created training_metadata.json
)

echo.
echo ========================================
echo Deployment setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file with your production settings
echo 2. Run: python app.py
echo 3. Navigate to http://localhost:5000
echo.
echo Default login: admin / change-this-secure-password
echo ========================================
pause
