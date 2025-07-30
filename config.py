import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fallback-secret-key'
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME') or 'admin'
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD') or 'admin123'
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # File storage settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    DATA_FOLDER = os.environ.get('DATA_FOLDER') or 'data'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16777216))  # 16MB
    
    # Face recognition settings
    FACE_RECOGNITION_TOLERANCE = 0.6
    MAX_FACE_DISTANCE = 0.4
    
    # CSV security settings
    MAX_CSV_ROWS = 10000
    ALLOWED_CSV_COLUMNS = [
        'student_id', 'name', 'email', 'department', 'year',
        'date', 'time', 'status', 'confidence'
    ]
