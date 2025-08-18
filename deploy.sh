#!/bin/bash

echo "========================================"
echo "Face Recognition System - Deployment Setup"
echo "========================================"

echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Creating necessary directories..."
mkdir -p uploads
mkdir -p data
mkdir -p static/css
mkdir -p static/js
mkdir -p static/images

echo ""
echo "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp ".env.example" ".env"
    echo "Please edit .env file with your production settings"
else
    echo ".env file already exists"
fi

echo ""
echo "Setting up data files..."
if [ ! -f "data/students.csv" ]; then
    echo "student_id,name,email,department,year,registration_date,photo_path" > data/students.csv
    echo "Created students.csv"
fi

if [ ! -f "data/attendance.csv" ]; then
    echo "student_id,name,date,time,status,confidence,method,latitude,longitude,location,city,state,country" > data/attendance.csv
    echo "Created attendance.csv"
fi

if [ ! -f "data/training_metadata.json" ]; then
    echo "{}" > data/training_metadata.json
    echo "Created training_metadata.json"
fi

echo ""
echo "========================================"
echo "Deployment setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your production settings"
echo "2. Run: python app.py"
echo "3. Navigate to http://localhost:5000"
echo ""
echo "Default login: admin / change-this-secure-password"
echo "========================================"
