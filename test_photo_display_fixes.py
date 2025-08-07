#!/usr/bin/env python3
"""
Comprehensive test script for photo display fixes
Tests all photo display functionality across the application
"""

import requests
import json
import os
import sys
from io import BytesIO
from PIL import Image, ImageDraw
import time

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

class PhotoDisplayTester:
    def __init__(self):
        self.session = requests.Session()
        self.logged_in = False
        self.test_student_id = f"TEST{int(time.time())}"
    
    def login(self):
        """Login to the system"""
        try:
            # Get login page first
            response = self.session.get(f"{BASE_URL}/login")
            if response.status_code != 200:
                print(f"❌ Cannot access login page: {response.status_code}")
                return False
            
            # Login
            login_data = {
                'username': TEST_USERNAME,
                'password': TEST_PASSWORD
            }
            
            response = self.session.post(f"{BASE_URL}/login", data=login_data)
            
            if response.status_code == 200 and ('dashboard' in response.url or 'students' in response.url):
                print("✅ Login successful")
                self.logged_in = True
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def create_test_image(self):
        """Create a test image for student photo"""
        img = Image.new('RGB', (200, 200), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face
        # Face circle
        draw.ellipse([50, 50, 150, 150], fill='peachpuff', outline='black')
        
        # Eyes
        draw.ellipse([70, 80, 80, 90], fill='black')
        draw.ellipse([120, 80, 130, 90], fill='black')
        
        # Mouth
        draw.arc([80, 110, 120, 130], 0, 180, fill='black')
        
        # Add text
        draw.text((60, 160), "TEST", fill='black')
        
        # Save to BytesIO
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        return img_buffer
    
    def test_student_registration_with_photo(self):
        """Test student registration with photo upload"""
        print("\n🧪 Testing: Student registration with photo")
        
        try:
            # Create test image
            test_image = self.create_test_image()
            
            student_data = {
                'student_id': self.test_student_id,
                'name': 'Test Photo Student',
                'email': f'test.photo.{self.test_student_id}@email.com',
                'department': 'Computer Science',
                'year': '2'
            }
            
            files = {'face_image': ('test_photo.jpg', test_image, 'image/jpeg')}
            response = self.session.post(f"{BASE_URL}/add_student", data=student_data, files=files)
            
            if response.status_code == 200 and 'successfully' in response.text.lower():
                print("✅ Student registration with photo successful")
                return True
            else:
                print(f"❌ Student registration failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in student registration test: {e}")
            return False
    
    def test_student_photo_endpoint(self):
        """Test the /student_photo/<student_id> endpoint"""
        print("\n🧪 Testing: Student photo endpoint")
        
        try:
            response = self.session.get(f"{BASE_URL}/student_photo/{self.test_student_id}")
            
            if response.status_code == 200:
                # Check if it's an image
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    print("✅ Student photo endpoint returns image")
                    return True
                else:
                    print(f"⚠️ Student photo endpoint returns non-image content: {content_type}")
                    return False
            elif response.status_code == 302:
                # Redirect to default avatar is acceptable
                print("✅ Student photo endpoint redirects to default avatar")
                return True
            else:
                print(f"❌ Student photo endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in photo endpoint test: {e}")
            return False
    
    def test_attendance_page_photos(self):
        """Test photo display in attendance page"""
        print("\n🧪 Testing: Photos in attendance page")
        
        try:
            response = self.session.get(f"{BASE_URL}/attendance")
            
            if response.status_code == 200:
                # Check if the page contains student photo references
                if 'student_photo' in response.text and 'img src' in response.text:
                    print("✅ Attendance page contains photo references")
                    return True
                else:
                    print("⚠️ Attendance page may not have photo references")
                    return False
            else:
                print(f"❌ Cannot access attendance page: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in attendance page test: {e}")
            return False
    
    def test_edit_student_page_photo(self):
        """Test photo display in edit student page"""
        print("\n🧪 Testing: Photo in edit student page")
        
        try:
            response = self.session.get(f"{BASE_URL}/edit_student/{self.test_student_id}")
            
            if response.status_code == 200:
                # Check if the page contains current photo display
                if 'current-student-photo' in response.text or 'student_photo' in response.text:
                    print("✅ Edit student page contains photo display")
                    return True
                else:
                    print("⚠️ Edit student page may not have photo display")
                    return False
            else:
                print(f"❌ Cannot access edit student page: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in edit student page test: {e}")
            return False
    
    def test_photo_recognition_results(self):
        """Test photo recognition results display"""
        print("\n🧪 Testing: Photo recognition results display")
        
        try:
            # Create test image for recognition
            test_image = self.create_test_image()
            
            files = {'photo': ('recognition_test.jpg', test_image, 'image/jpeg')}
            response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Recognition response: {json.dumps(result, indent=2)}")
                
                if result.get('success'):
                    # Check if response has the expected structure
                    expected_keys = ['faces_detected', 'faces_recognized', 'data']
                    if all(key in result for key in expected_keys):
                        print("✅ Photo recognition endpoint returns proper structure")
                        return True
                    else:
                        print("⚠️ Photo recognition response missing expected keys")
                        return False
                else:
                    print(f"⚠️ Photo recognition failed: {result.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Photo recognition endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in photo recognition test: {e}")
            return False
    
    def test_default_avatar_fallback(self):
        """Test default avatar fallback for non-existent student"""
        print("\n🧪 Testing: Default avatar fallback")
        
        try:
            # Test with non-existent student ID
            response = self.session.get(f"{BASE_URL}/student_photo/NONEXISTENT")
            
            if response.status_code == 302:
                # Should redirect to default avatar
                print("✅ Default avatar fallback working")
                return True
            elif response.status_code == 200:
                # Might be serving default avatar directly
                print("✅ Default avatar served directly")
                return True
            else:
                print(f"❌ Default avatar fallback failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in default avatar test: {e}")
            return False
    
    def cleanup_test_data(self):
        """Clean up test student data"""
        print("\n🧹 Cleaning up test data...")
        
        try:
            response = self.session.get(f"{BASE_URL}/delete_student/{self.test_student_id}")
            if response.status_code in [200, 302]:
                print("✅ Test student deleted")
            else:
                print(f"⚠️ Could not delete test student: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error cleaning up: {e}")
    
    def run_all_tests(self):
        """Run all photo display tests"""
        print("🚀 Starting Photo Display Tests")
        print("=" * 60)
        
        if not self.login():
            print("❌ Cannot proceed without login")
            return False
        
        tests = [
            self.test_student_registration_with_photo,
            self.test_student_photo_endpoint,
            self.test_attendance_page_photos,
            self.test_edit_student_page_photo,
            self.test_photo_recognition_results,
            self.test_default_avatar_fallback
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        # Cleanup
        self.cleanup_test_data()
        
        print("\n" + "=" * 60)
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All photo display tests PASSED!")
            return True
        else:
            print("⚠️ Some tests failed. Please check the implementation.")
            return False

def main():
    """Main function"""
    tester = PhotoDisplayTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
