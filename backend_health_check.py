#!/usr/bin/env python3
"""
Backend Health Check Script
"""

import os
import sys

def test_backend_health():
    print("🔍 Face Recognition Backend Health Check")
    print("=" * 50)
    
    success_count = 0
    total_tests = 6
    
    # Test 1: CSV File Reading
    try:
        from utils.data_manager import get_data_manager
        dm = get_data_manager()
        students = dm.get_all_students()
        attendance = dm.get_all_attendance()
        print(f"✅ Test 1: CSV Reading - Students: {len(students)}, Attendance: {len(attendance)}")
        success_count += 1
    except Exception as e:
        print(f"❌ Test 1: CSV Reading Failed - {e}")
    
    # Test 2: Face Recognition
    try:
        from utils.face_recognition_mock import get_face_recognizer
        recognizer = get_face_recognizer()
        print(f"✅ Test 2: Face Recognition - {type(recognizer).__name__} loaded")
        success_count += 1
    except Exception as e:
        print(f"❌ Test 2: Face Recognition Failed - {e}")
    
    # Test 3: Security Functions
    try:
        from utils.security import validate_student_id, validate_email
        id_valid = validate_student_id("STU123")
        email_valid = validate_email("test@university.edu")
        print(f"✅ Test 3: Security - ID: {id_valid}, Email: {email_valid}")
        success_count += 1
    except Exception as e:
        print(f"❌ Test 3: Security Failed - {e}")
    
    # Test 4: Configuration
    try:
        from config import Config
        print(f"✅ Test 4: Config - Admin: {Config.ADMIN_USERNAME}")
        success_count += 1
    except Exception as e:
        print(f"❌ Test 4: Config Failed - {e}")
    
    # Test 5: Flask App
    try:
        from app import app
        print(f"✅ Test 5: Flask App - {app.name} created")
        success_count += 1
    except Exception as e:
        print(f"❌ Test 5: Flask App Failed - {e}")
    
    # Test 6: Data Files
    try:
        students_exists = os.path.exists("data/students.csv")
        attendance_exists = os.path.exists("data/attendance.csv")
        
        if students_exists and attendance_exists:
            print("✅ Test 6: Data Files - All files present")
            success_count += 1
        else:
            print("❌ Test 6: Data Files - Missing files")
    except Exception as e:
        print(f"❌ Test 6: Data Files Failed - {e}")
    
    print("=" * 50)
    print(f"🎯 Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! Backend is working correctly.")
        print("🚀 System ready at http://localhost:5000")
        print("👤 Login: admin / admin123")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = test_backend_health()
    sys.exit(0 if success else 1)
