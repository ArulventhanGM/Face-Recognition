#!/usr/bin/env python3
"""
Production Readiness Verification Script
Checks if the Face Recognition System is ready for deployment
"""
import os
import sys
from pathlib import Path
import importlib.util

def check_file_structure():
    """Verify all required files and directories exist"""
    print("🔍 Checking file structure...")
    
    required_files = [
        'app.py', 'config.py', 'requirements.txt', 'README.md',
        'LICENSE', '.env.example', '.gitignore'
    ]
    
    required_dirs = [
        'data', 'uploads', 'static', 'templates', 'utils',
        'static/css', 'static/js', 'static/images'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"❌ Missing files: {missing_files}")
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ File structure complete")
    return True

def check_python_imports():
    """Verify all required Python modules can be imported"""
    print("🔍 Checking Python dependencies...")
    
    required_modules = [
        'flask', 'opencv-python', 'numpy', 'pandas', 'PIL',
        'exifread', 'geopy', 'requests'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            # Handle special module names
            import_name = module
            if module == 'opencv-python':
                import_name = 'cv2'
            elif module == 'PIL':
                import_name = 'PIL'
            
            importlib.util.find_spec(import_name)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing modules: {missing_modules}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def check_environment_config():
    """Check environment configuration"""
    print("🔍 Checking environment configuration...")
    
    if not Path('.env').exists():
        print("⚠️  No .env file found (will use .env.example)")
    
    # Check critical config values
    from config import Config
    
    issues = []
    
    # Check secret key
    if Config.SECRET_KEY in ['your-secret-key-change-this-in-production', 'dev-secret-key']:
        issues.append("SECRET_KEY should be changed for production")
    
    # Check admin password
    if Config.ADMIN_PASSWORD in ['admin123', 'change-this-secure-password']:
        issues.append("ADMIN_PASSWORD should be changed for production")
    
    if issues:
        print("⚠️  Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Environment configuration ready")
    return True

def check_application_health():
    """Basic application health check"""
    print("🔍 Checking application health...")
    
    try:
        # Try importing the main app
        from app import app
        print("✅ Flask application imports successfully")
        
        # Try importing utilities
        from utils.data_manager import get_data_manager
        from utils.geo_location import get_geo_location_extractor
        
        data_manager = get_data_manager()
        geo_extractor = get_geo_location_extractor()
        
        print("✅ Core utilities initialize successfully")
        return True
        
    except Exception as e:
        print(f"❌ Application health check failed: {e}")
        return False

def check_security_settings():
    """Check security configuration"""
    print("🔍 Checking security settings...")
    
    from config import Config
    
    security_checks = []
    
    # Check debug mode
    if Config.DEBUG:
        security_checks.append("DEBUG mode is enabled (should be False for production)")
    
    # Check allowed file extensions
    if not hasattr(Config, 'ALLOWED_EXTENSIONS'):
        security_checks.append("ALLOWED_EXTENSIONS not configured")
    
    if security_checks:
        print("⚠️  Security concerns:")
        for check in security_checks:
            print(f"   - {check}")
        return False
    
    print("✅ Security settings verified")
    return True

def generate_deployment_summary():
    """Generate deployment summary"""
    print("\n" + "="*60)
    print("📋 DEPLOYMENT SUMMARY")
    print("="*60)
    
    print("🚀 Deployment Options:")
    print("   1. Direct: python start.py")
    print("   2. Docker: docker-compose up -d")
    print("   3. Scripts: ./deploy.sh or deploy.bat")
    
    print("\n🌐 Access Information:")
    print("   URL: http://localhost:5000")
    print("   Admin: Check .env file for credentials")
    
    print("\n📁 Important Directories:")
    print("   - data/: Student and attendance records")
    print("   - uploads/: Temporary file storage")
    print("   - static/: Web assets (CSS, JS, images)")
    
    print("\n🔒 Security Reminders:")
    print("   - Change default admin password")
    print("   - Use HTTPS in production")
    print("   - Regular backups of data/ directory")
    
    print("\n📚 Documentation:")
    print("   - README.md: Complete usage guide")
    print("   - DEPLOYMENT.md: Deployment checklist")
    print("   - LICENSE: MIT License terms")
    
    print("="*60)

def main():
    """Main verification function"""
    print("🎯 Face Recognition System - Production Verification")
    print("="*60)
    
    checks = [
        check_file_structure,
        check_python_imports,
        check_environment_config,
        check_security_settings,
        check_application_health
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check in checks:
        try:
            if check():
                passed_checks += 1
            print()  # Empty line for readability
        except Exception as e:
            print(f"❌ Check failed with error: {e}\n")
    
    print("="*60)
    print(f"📊 VERIFICATION RESULTS: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 System is ready for production deployment!")
        generate_deployment_summary()
        return True
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        print("💡 Run this script again after fixing the issues.")
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
