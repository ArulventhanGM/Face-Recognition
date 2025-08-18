#!/usr/bin/env python3
"""
Face Recognition System - Production Startup Script
"""
import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher required")
        return False
    
    # Check required directories
    required_dirs = ['data', 'uploads', 'static', 'templates', 'utils']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"ERROR: Required directory '{dir_name}' not found")
            return False
    
    # Check required files
    required_files = ['app.py', 'requirements.txt', '.env']
    for file_name in required_files:
        if not Path(file_name).exists():
            print(f"ERROR: Required file '{file_name}' not found")
            return False
    
    print("✓ System requirements check passed")
    return True

def setup_environment():
    """Setup environment for production"""
    print("Setting up production environment...")
    
    # Create .env from example if it doesn't exist
    if not Path('.env').exists():
        if Path('.env.example').exists():
            import shutil
            shutil.copy('.env.example', '.env')
            print("Created .env file from template")
            print("WARNING: Please edit .env file with your production settings!")
        else:
            print("ERROR: No .env or .env.example file found")
            return False
    
    # Set production environment
    os.environ['FLASK_ENV'] = 'production'
    os.environ['PYTHONPATH'] = os.getcwd()
    
    print("✓ Environment setup complete")
    return True

def main():
    """Main startup function"""
    print("="*50)
    print("Face Recognition System - Starting Up")
    print("="*50)
    
    if not check_requirements():
        print("\nStartup failed: Requirements check failed")
        sys.exit(1)
    
    if not setup_environment():
        print("\nStartup failed: Environment setup failed")  
        sys.exit(1)
    
    print("\nStarting Flask application...")
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print("-"*50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print(f"\nError starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
