#!/usr/bin/env python3
"""
Main entry point for Google Cloud Platform deployment
Face Recognition Application
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app import app

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # For production deployment (Google Cloud)
    # The app object is imported and used by the WSGI server
    application = app
