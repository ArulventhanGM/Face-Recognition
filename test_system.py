#!/usr/bin/env python3
"""
Test script for Face Recognition Academic System
"""

import os
import sys
import unittest
import tempfile
import shutil

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestFaceRecognitionSystem(unittest.TestCase):
    """Test cases for the face recognition system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        os.environ['DATA_FOLDER'] = self.test_dir
        os.environ['UPLOAD_FOLDER'] = os.path.join(self.test_dir, 'uploads')
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_config_import(self):
        """Test that config can be imported"""
        try:
            from config import Config
            self.assertTrue(hasattr(Config, 'SECRET_KEY'))
            self.assertTrue(hasattr(Config, 'ADMIN_USERNAME'))
            print("✓ Config import successful")
        except ImportError as e:
            self.fail(f"Failed to import config: {e}")
    
    def test_security_utils(self):
        """Test security utilities"""
        try:
            from utils.security import validate_student_id, validate_email, sanitize_filename
            
            # Test student ID validation
            self.assertTrue(validate_student_id("STU001"))
            self.assertFalse(validate_student_id(""))
            self.assertFalse(validate_student_id("@#$"))
            
            # Test email validation
            self.assertTrue(validate_email("test@example.com"))
            self.assertFalse(validate_email("invalid-email"))
            
            # Test filename sanitization
            safe_name = sanitize_filename("test<>file.jpg")
            self.assertNotIn("<", safe_name)
            self.assertNotIn(">", safe_name)
            
            print("✓ Security utils tests passed")
        except ImportError as e:
            self.fail(f"Failed to import security utils: {e}")
    
    def test_face_recognition_import(self):
        """Test face recognition utilities (mock version)"""
        try:
            from utils.face_recognition_mock import MockFaceRecognitionSystem
            
            face_system = MockFaceRecognitionSystem()
            self.assertIsNotNone(face_system)
            
            print("✓ Face recognition (mock) import successful")
        except ImportError as e:
            self.fail(f"Failed to import face recognition utils: {e}")
    
    def test_data_manager_import(self):
        """Test data manager import"""
        try:
            from utils.data_manager import DataManager
            
            data_manager = DataManager(self.test_dir)
            self.assertIsNotNone(data_manager)
            self.assertEqual(data_manager.data_folder, self.test_dir)
            
            print("✓ Data manager import successful")
        except ImportError as e:
            self.fail(f"Failed to import data manager: {e}")
    
    def test_flask_app_creation(self):
        """Test Flask app creation"""
        try:
            from app import app
            
            self.assertIsNotNone(app)
            self.assertEqual(app.name, 'app')
            
            # Test that routes are registered
            rule_names = [rule.rule for rule in app.url_map.iter_rules()]
            expected_routes = ['/', '/login', '/dashboard', '/students']
            
            for route in expected_routes:
                self.assertIn(route, rule_names)
            
            print("✓ Flask app creation successful")
        except ImportError as e:
            self.fail(f"Failed to import Flask app: {e}")

def run_tests():
    """Run all tests"""
    print("Face Recognition Academic System - Test Suite")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check required packages
    required_packages = [
        ('flask', 'flask'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('pillow', 'PIL'),
        ('werkzeug', 'werkzeug'),
        ('flask_login', 'flask_login')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} available")
        except ImportError:
            if package_name not in ['opencv-python']:  # Optional for basic testing
                missing_packages.append(package_name)
            print(f"❌ {package_name} missing")
    
    if missing_packages:
        print(f"\n❌ Missing critical packages: {', '.join(missing_packages)}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    print("\n" + "=" * 50)
    print("Running unit tests...")
    print("=" * 50)
    
    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFaceRecognitionSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed! System is ready to run.")
        print("\nTo start the application:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Login: admin / admin123")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
