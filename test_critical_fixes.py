#!/usr/bin/env python3
"""
Test script to verify all critical fixes are working
"""

import requests
import json
import os
import sys
import time
from io import BytesIO
from PIL import Image, ImageDraw

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

class CriticalFixesTester:
    def __init__(self):
        self.session = requests.Session()
        self.logged_in = False
        self.test_student_id = f"FIX{int(time.time())}"
    
    def login(self):
        """Login to the system"""
        try:
            response = self.session.get(f"{BASE_URL}/login")
            if response.status_code != 200:
                print(f"❌ Cannot access login page: {response.status_code}")
                return False
            
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
    
    def create_test_image(self, unique_id=""):
        """Create a unique test image"""
        img = Image.new('RGB', (300, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face
        draw.ellipse([75, 75, 225, 225], fill='peachpuff', outline='black')
        draw.ellipse([110, 120, 130, 140], fill='black')  # Left eye
        draw.ellipse([170, 120, 190, 140], fill='black')  # Right eye
        draw.arc([125, 160, 175, 190], 0, 180, fill='black')  # Mouth
        draw.text((120, 240), unique_id, fill='black')
        
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer
    
    def test_student_crud_operations(self):
        """Test complete CRUD operations"""
        print("\n🔧 TESTING STUDENT CRUD OPERATIONS")
        print("=" * 50)
        
        # CREATE
        print("\n1️⃣ Testing Student Creation...")
        test_image = self.create_test_image(self.test_student_id)
        
        student_data = {
            'student_id': self.test_student_id,
            'name': 'CRUD Test Student',
            'email': f'crud.test.{self.test_student_id}@email.com',
            'department': 'Computer Science',
            'year': '3'
        }
        
        files = {'face_image': ('crud_test.jpg', test_image, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/add_student", data=student_data, files=files)
        
        if response.status_code == 200 and 'successfully' in response.text.lower():
            print("✅ CREATE: Student creation working")
        else:
            print(f"❌ CREATE: Failed ({response.status_code})")
            return False
        
        # READ
        print("\n2️⃣ Testing Student Read...")
        response = self.session.get(f"{BASE_URL}/students")
        
        if response.status_code == 200 and self.test_student_id in response.text:
            print("✅ READ: Student listing working")
        else:
            print("❌ READ: Failed")
            return False
        
        # UPDATE
        print("\n3️⃣ Testing Student Update...")
        updated_data = {
            'name': 'Updated CRUD Student',
            'email': f'updated.crud.{self.test_student_id}@email.com',
            'department': 'Information Technology',
            'year': '4'
        }
        
        response = self.session.post(f"{BASE_URL}/edit_student/{self.test_student_id}", data=updated_data)
        
        if response.status_code == 200:
            # Verify update by checking students page
            verify_response = self.session.get(f"{BASE_URL}/students")
            if 'Updated CRUD Student' in verify_response.text:
                print("✅ UPDATE: Student update working")
            else:
                print("❌ UPDATE: Update not reflected")
                return False
        else:
            print(f"❌ UPDATE: Failed ({response.status_code})")
            return False
        
        # DELETE
        print("\n4️⃣ Testing Student Delete...")
        response = self.session.get(f"{BASE_URL}/delete_student/{self.test_student_id}")
        
        if response.status_code in [200, 302]:
            # Verify deletion
            verify_response = self.session.get(f"{BASE_URL}/students")
            if self.test_student_id not in verify_response.text:
                print("✅ DELETE: Student deletion working")
                return True
            else:
                print("❌ DELETE: Student still exists after deletion")
                return False
        else:
            print(f"❌ DELETE: Failed ({response.status_code})")
            return False
    
    def test_photo_recognition_workflow(self):
        """Test complete photo recognition workflow"""
        print("\n📸 TESTING PHOTO RECOGNITION WORKFLOW")
        print("=" * 50)
        
        # Create test student for recognition
        print("\n1️⃣ Creating test student...")
        unique_id = f"REC{int(time.time())}"
        test_image = self.create_test_image(unique_id)
        
        student_data = {
            'student_id': unique_id,
            'name': 'Recognition Test Student',
            'email': f'recog.{unique_id}@email.com',
            'department': 'Engineering',
            'year': '2'
        }
        
        files = {'face_image': ('recog_test.jpg', test_image, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/add_student", data=student_data, files=files)
        
        if response.status_code != 200 or 'successfully' not in response.text.lower():
            print("❌ Failed to create test student")
            return False
        
        print("✅ Test student created")
        
        # Wait for processing
        time.sleep(1)
        
        # Test recognition with same image
        print("\n2️⃣ Testing photo recognition...")
        test_image_2 = self.create_test_image(unique_id)  # Same image
        
        files = {'photo': ('recognition_test.jpg', test_image_2, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        success = False
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                
                # Check response structure
                required_keys = ['success', 'faces_detected', 'faces_recognized', 'data']
                if all(key in result for key in required_keys):
                    print("✅ Response structure: Correct")
                    
                    if result.get('success'):
                        if result.get('faces_detected') > 0:
                            print(f"✅ Face detection: {result['faces_detected']} face(s) detected")
                            
                            if result.get('faces_recognized') > 0:
                                recognized = result['data'][0]
                                if recognized['student_id'] == unique_id:
                                    confidence = recognized.get('confidence', 0)
                                    print(f"✅ Face recognition: Correctly identified with {confidence:.1%} confidence")
                                    success = True
                                else:
                                    print(f"❌ Face recognition: Wrong student identified")
                            else:
                                print("⚠️ Face recognition: No faces recognized")
                        else:
                            print("❌ Face detection: No faces detected")
                    else:
                        print(f"❌ Recognition failed: {result.get('message', 'Unknown error')}")
                else:
                    print("❌ Response structure: Missing required keys")
                    
            except json.JSONDecodeError:
                print("❌ Invalid JSON response")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
        
        # Cleanup
        self.session.get(f"{BASE_URL}/delete_student/{unique_id}")
        
        return success
    
    def test_photo_display_functionality(self):
        """Test photo display across different pages"""
        print("\n🖼️ TESTING PHOTO DISPLAY FUNCTIONALITY")
        print("=" * 50)
        
        # Create test student with photo
        print("\n1️⃣ Creating student with photo...")
        unique_id = f"PHOTO{int(time.time())}"
        test_image = self.create_test_image(unique_id)
        
        student_data = {
            'student_id': unique_id,
            'name': 'Photo Display Test',
            'email': f'photo.{unique_id}@email.com',
            'department': 'Design',
            'year': '1'
        }
        
        files = {'face_image': ('photo_test.jpg', test_image, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/add_student", data=student_data, files=files)
        
        if response.status_code != 200 or 'successfully' not in response.text.lower():
            print("❌ Failed to create test student")
            return False
        
        print("✅ Test student created")
        
        # Test photo endpoint
        print("\n2️⃣ Testing photo endpoint...")
        response = self.session.get(f"{BASE_URL}/student_photo/{unique_id}")
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                print("✅ Photo endpoint: Returns image")
                photo_working = True
            else:
                print(f"⚠️ Photo endpoint: Returns {content_type}")
                photo_working = False
        elif response.status_code == 302:
            print("✅ Photo endpoint: Redirects to default avatar")
            photo_working = True
        else:
            print(f"❌ Photo endpoint: Failed ({response.status_code})")
            photo_working = False
        
        # Test attendance page photo display
        print("\n3️⃣ Testing attendance page photos...")
        response = self.session.get(f"{BASE_URL}/attendance")
        
        if response.status_code == 200 and 'student_photo' in response.text:
            print("✅ Attendance page: Contains photo references")
            attendance_photos = True
        else:
            print("❌ Attendance page: Missing photo references")
            attendance_photos = False
        
        # Test edit student page photo display
        print("\n4️⃣ Testing edit student page photo...")
        response = self.session.get(f"{BASE_URL}/edit_student/{unique_id}")
        
        if response.status_code == 200 and ('current-student-photo' in response.text or 'student_photo' in response.text):
            print("✅ Edit student page: Contains photo display")
            edit_photos = True
        else:
            print("❌ Edit student page: Missing photo display")
            edit_photos = False
        
        # Cleanup
        self.session.get(f"{BASE_URL}/delete_student/{unique_id}")
        
        return photo_working and attendance_photos and edit_photos
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 TESTING ALL CRITICAL FIXES")
        print("=" * 60)
        
        if not self.login():
            print("❌ Cannot proceed without login")
            return False
        
        tests = [
            ("Student CRUD Operations", self.test_student_crud_operations),
            ("Photo Recognition Workflow", self.test_photo_recognition_workflow),
            ("Photo Display Functionality", self.test_photo_display_functionality)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            results[test_name] = test_func()
        
        print("\n" + "=" * 60)
        print("📊 FINAL TEST RESULTS")
        print("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        passed = sum(results.values())
        total = len(results)
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL CRITICAL FIXES ARE WORKING!")
        else:
            print("⚠️ Some issues remain. Check the details above.")
        
        return passed == total

def main():
    """Main function"""
    tester = CriticalFixesTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
