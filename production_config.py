"""
Production configuration for Face Recognition System
"""
import os
from datetime import timedelta

class ProductionConfig:
    """Production configuration class"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key-change-this'
    
    # Admin credentials
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME') or 'admin'
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD') or 'change-this-secure-password'
    
    # Flask settings
    DEBUG = False
    TESTING = False
    
    # File uploads
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH') or 16777216)  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    # Data storage
    DATA_FOLDER = os.environ.get('DATA_FOLDER') or 'data'
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    SESSION_COOKIE_SECURE = True  # Enable for HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Security headers
    SEND_FILE_MAX_AGE_DEFAULT = timedelta(hours=1)
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'face_recognition.log'
    
    # Face recognition settings
    FACE_RECOGNITION_THRESHOLD = 0.6
    MAX_FACES_PER_IMAGE = 50
    
    # Location services
    ENABLE_LOCATION_EXTRACTION = True
    GEOCODING_TIMEOUT = 10
    
    @staticmethod
    def init_app(app):
        """Initialize application with production config"""
        # Create required directories
        os.makedirs(ProductionConfig.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(ProductionConfig.DATA_FOLDER, exist_ok=True)
        
        # Setup logging
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug and not app.testing:
            # File logging
            file_handler = RotatingFileHandler(
                ProductionConfig.LOG_FILE,
                maxBytes=10240000,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            
            app.logger.setLevel(logging.INFO)
            app.logger.info('Face Recognition System startup')


class DevelopmentConfig:
    """Development configuration class"""
    
    # Security
    SECRET_KEY = 'dev-secret-key'
    
    # Admin credentials
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = 'admin123'
    
    # Flask settings
    DEBUG = True
    TESTING = False
    
    # File uploads
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16777216  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    # Data storage
    DATA_FOLDER = 'data'
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Face recognition settings
    FACE_RECOGNITION_THRESHOLD = 0.6
    MAX_FACES_PER_IMAGE = 50
    
    # Location services
    ENABLE_LOCATION_EXTRACTION = True
    GEOCODING_TIMEOUT = 10
    
    @staticmethod
    def init_app(app):
        """Initialize application with development config"""
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('data', exist_ok=True)


# Configuration selection
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
