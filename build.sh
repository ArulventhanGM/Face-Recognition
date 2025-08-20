#!/usr/bin/env bash
# Build script for Render deployment

echo "Starting build process..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data
mkdir -p uploads
mkdir -p static/images
mkdir -p static/temp

# Initialize data files if they don't exist
echo "Setting up default data files..."

# Create students.csv if it doesn't exist
if [ ! -f data/students.csv ]; then
    echo "student_id,name,email,department,academic_year,registration_date,photo_path" > data/students.csv
    echo "Created students.csv"
fi

# Create attendance.csv if it doesn't exist  
if [ ! -f data/attendance.csv ]; then
    echo "student_id,name,date,time,status,confidence,method,latitude,longitude,location,city,state,country" > data/attendance.csv
    echo "Created attendance.csv"
fi

# Create training metadata if it doesn't exist
if [ ! -f data/training_metadata.json ]; then
    echo "{}" > data/training_metadata.json
    echo "Created training_metadata.json"
fi

# Create face embeddings file if it doesn't exist
if [ ! -f data/face_embeddings.pkl ]; then
    python -c "import pickle; pickle.dump({}, open('data/face_embeddings.pkl', 'wb'))"
    echo "Created face_embeddings.pkl"
fi

echo "Build process completed successfully!"
