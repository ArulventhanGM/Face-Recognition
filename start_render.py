#!/usr/bin/env python3
"""
Render startup script for Face Recognition System
"""
import os
import sys

if __name__ == "__main__":
    # Set environment variables if not already set
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('DEBUG', 'False')
    
    # Import and run the application
    from app import app
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get('PORT', 10000))
    
    print(f"🚀 Starting Face Recognition System on port {port}")
    print(f"🌐 Environment: {os.environ.get('FLASK_ENV', 'production')}")
    print(f"🔧 Debug mode: {os.environ.get('DEBUG', 'False')}")
    
    # Run the application
    app.run(host='0.0.0.0', port=port, debug=False)
