#!/bin/bash

# Google Cloud Build script for Face Recognition App

echo "Starting Google Cloud Build..."

# Install system dependencies
apt-get update
apt-get install -y python3-opencv

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p data uploads static/css static/js static/images

# Set permissions
chmod -R 755 static
chmod -R 755 templates
chmod -R 755 utils

# Initialize data files if they don't exist
python3 -c "
from utils.data_manager import get_data_manager
dm = get_data_manager()
print('Data manager initialized successfully')
"

echo "Build completed successfully!"
