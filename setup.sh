#!/bin/bash

echo "Setting up Face Recognition Academic System..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH."
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "Python found!"
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install requirements."
    exit 1
fi

# Copy .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file with your configuration."
fi

echo
echo "Setup complete!"
echo
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the application: python app.py"
echo "3. Open browser to: http://localhost:5000"
echo "4. Login with: admin / admin123 (change in .env)"
echo
