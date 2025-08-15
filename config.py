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
    
    # Face recognition settings - Optimized for 90%+ accuracy
    FACE_RECOGNITION_TOLERANCE = 0.65  # Optimized threshold for enhanced backend
    MAX_FACE_DISTANCE = 0.35           # Stricter distance for better accuracy
    MIN_FACE_SIZE = 30                 # Minimum face size for detection
    MAX_FACE_SIZE = 300               # Maximum face size for detection
    FACE_PADDING_RATIO = 0.2          # Padding around detected faces
    CONFIDENCE_THRESHOLD = 0.7        # Face detection confidence threshold
    
    # CSV security settings
    MAX_CSV_ROWS = 10000
    ALLOWED_CSV_COLUMNS = [
        'student_id', 'name', 'email', 'department', 'year',
        'date', 'time', 'status', 'confidence'
    ]
