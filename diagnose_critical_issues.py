#!/usr/bin/env python3
"""
Comprehensive diagnostic script for critical Face Recognition System issues
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

class CriticalIssuesDiagnostic:
    def __init__(self):
        self.session = requests.Session()
        self.logged_in = False
        self.test_student_id = f"DIAG{int(time.time())}"
    
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
    
    def create_test_image(self):
        """Create a test image for student photo"""
        img = Image.new('RGB', (300, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face
        draw.ellipse([75, 75, 225, 225], fill='peachpuff', outline='black')
        draw.ellipse([110, 120, 130, 140], fill='black')  # Left eye
        draw.ellipse([170, 120, 190, 140], fill='black')  # Right eye
        draw.arc([125, 160, 175, 190], 0, 180, fill='black')  # Mouth
        draw.text((120, 240), "DIAG", fill='black')
        
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer
    
    def test_student_crud_operations(self):
        """Test student CRUD operations"""
        print("\n🔧 TESTING STUDENT CRUD OPERATIONS")
        print("=" * 50)
        
        # Test 1: Create student
        print("\n1️⃣ Testing Student Creation...")
        test_image = self.create_test_image()
        
        student_data = {
            'student_id': self.test_student_id,
            'name': 'Diagnostic Test Student',
            'email': f'diag.test.{self.test_student_id}@email.com',
            'department': 'Computer Science',
            'year': '3'
        }
        
        files = {'face_image': ('diag_test.jpg', test_image, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/add_student", data=student_data, files=files)
        
        if response.status_code == 200 and 'successfully' in response.text.lower():
            print("✅ Student creation: WORKING")
            student_created = True
        else:
            print(f"❌ Student creation: FAILED ({response.status_code})")
            student_created = False
        
        if not student_created:
            return False
        
        # Test 2: Read student (verify creation)
        print("\n2️⃣ Testing Student Read...")
        response = self.session.get(f"{BASE_URL}/students")
        
        if response.status_code == 200 and self.test_student_id in response.text:
            print("✅ Student read: WORKING")
        else:
            print("❌ Student read: FAILED")
        
        # Test 3: Update student
        print("\n3️⃣ Testing Student Update...")
        updated_data = {
            'name': 'Updated Diagnostic Student',
            'email': f'updated.diag.{self.test_student_id}@email.com',
            'department': 'Information Technology',
            'year': '4'
        }
        
        response = self.session.post(f"{BASE_URL}/edit_student/{self.test_student_id}", data=updated_data)
        
        if response.status_code == 200:
            # Check if update was successful by looking for success message or redirect
            if 'successfully' in response.text.lower() or response.url.endswith('/students'):
                print("✅ Student update: WORKING")
            else:
                print("❌ Student update: FAILED (no success confirmation)")
        else:
            print(f"❌ Student update: FAILED ({response.status_code})")
        
        # Test 4: Delete student
        print("\n4️⃣ Testing Student Delete...")
        response = self.session.get(f"{BASE_URL}/delete_student/{self.test_student_id}")
        
        if response.status_code == 200 or response.status_code == 302:
            # Check if student is actually deleted
            verify_response = self.session.get(f"{BASE_URL}/students")
            if self.test_student_id not in verify_response.text:
                print("✅ Student delete: WORKING")
                return True
            else:
                print("❌ Student delete: FAILED (student still exists)")
                return False
        else:
            print(f"❌ Student delete: FAILED ({response.status_code})")
            return False
    
    def test_photo_recognition_display(self):
        """Test photo recognition results display"""
        print("\n📸 TESTING PHOTO RECOGNITION DISPLAY")
        print("=" * 50)
        
        # First, create a test student for recognition
        print("\n1️⃣ Creating test student for recognition...")
        test_image = self.create_test_image()
        
        student_data = {
            'student_id': self.test_student_id,
            'name': 'Recognition Test Student',
            'email': f'recog.test.{self.test_student_id}@email.com',
            'department': 'Computer Science',
            'year': '2'
        }
        
        files = {'face_image': ('recog_test.jpg', test_image, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/add_student", data=student_data, files=files)
        
        if response.status_code != 200 or 'successfully' not in response.text.lower():
            print("❌ Failed to create test student for recognition")
            return False
        
        print("✅ Test student created")
        
        # Test recognition endpoint
        print("\n2️⃣ Testing recognition endpoint...")
        test_image_2 = self.create_test_image()  # Same image for recognition
        
        files = {'photo': ('recognition_test.jpg', test_image_2, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response Structure: {json.dumps(result, indent=2)}")
                
                # Check response structure
                required_keys = ['success', 'faces_detected', 'faces_recognized', 'data']
                missing_keys = [key for key in required_keys if key not in result]
                
                if missing_keys:
                    print(f"❌ Recognition endpoint: Missing keys {missing_keys}")
                    return False
                
                if result.get('success'):
                    print("✅ Recognition endpoint: WORKING")
                    print(f"   Faces detected: {result.get('faces_detected', 0)}")
                    print(f"   Faces recognized: {result.get('faces_recognized', 0)}")
                    return True
                else:
                    print(f"❌ Recognition endpoint: Failed - {result.get('message', 'Unknown error')}")
                    return False
                    
            except json.JSONDecodeError:
                print(f"❌ Recognition endpoint: Invalid JSON response")
                print(f"Response text: {response.text[:200]}")
                return False
        else:
            print(f"❌ Recognition endpoint: HTTP Error {response.status_code}")
            return False
    
    def test_face_recognition_accuracy(self):
        """Test face recognition accuracy with same image"""
        print("\n🎯 TESTING FACE RECOGNITION ACCURACY")
        print("=" * 50)
        
        # Create a unique test image
        print("\n1️⃣ Creating unique test student...")
        img = Image.new('RGB', (400, 400), color='lightgreen')
        draw = ImageDraw.Draw(img)
        
        # Draw a distinctive face
        draw.ellipse([100, 100, 300, 300], fill='tan', outline='black', width=3)
        draw.ellipse([140, 160, 170, 190], fill='blue')  # Left eye
        draw.ellipse([230, 160, 260, 190], fill='blue')  # Right eye
        draw.ellipse([190, 200, 210, 220], fill='black')  # Nose
        draw.arc([160, 240, 240, 280], 0, 180, fill='red', width=3)  # Mouth
        draw.text((170, 320), "UNIQUE", fill='black')
        
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Register student with this image
        unique_student_id = f"UNIQUE{int(time.time())}"
        student_data = {
            'student_id': unique_student_id,
            'name': 'Unique Test Student',
            'email': f'unique.test.{unique_student_id}@email.com',
            'department': 'Engineering',
            'year': '1'
        }
        
        files = {'face_image': ('unique_test.jpg', img_buffer, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/add_student", data=student_data, files=files)
        
        if response.status_code != 200 or 'successfully' not in response.text.lower():
            print("❌ Failed to create unique test student")
            return False
        
        print("✅ Unique test student created")
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Test recognition with the EXACT same image
        print("\n2️⃣ Testing recognition with exact same image...")
        
        # Recreate the exact same image
        img2 = Image.new('RGB', (400, 400), color='lightgreen')
        draw2 = ImageDraw.Draw(img2)
        draw2.ellipse([100, 100, 300, 300], fill='tan', outline='black', width=3)
        draw2.ellipse([140, 160, 170, 190], fill='blue')
        draw2.ellipse([230, 160, 260, 190], fill='blue')
        draw2.ellipse([190, 200, 210, 220], fill='black')
        draw2.arc([160, 240, 240, 280], 0, 180, fill='red', width=3)
        draw2.text((170, 320), "UNIQUE", fill='black')
        
        img_buffer2 = BytesIO()
        img2.save(img_buffer2, format='JPEG')
        img_buffer2.seek(0)
        
        files = {'photo': ('recognition_exact.jpg', img_buffer2, 'image/jpeg')}
        response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Recognition Result: {json.dumps(result, indent=2)}")
                
                if result.get('success') and result.get('faces_detected') > 0:
                    if result.get('faces_recognized') > 0:
                        recognized_student = result['data'][0]
                        if recognized_student['student_id'] == unique_student_id:
                            confidence = recognized_student.get('confidence', 0)
                            print(f"✅ Face recognition accuracy: WORKING")
                            print(f"   Correctly identified: {unique_student_id}")
                            print(f"   Confidence: {confidence:.2%}")
                            
                            # Cleanup
                            self.session.get(f"{BASE_URL}/delete_student/{unique_student_id}")
                            return True
                        else:
                            print(f"❌ Face recognition accuracy: WRONG MATCH")
                            print(f"   Expected: {unique_student_id}")
                            print(f"   Got: {recognized_student['student_id']}")
                    else:
                        print("❌ Face recognition accuracy: NO RECOGNITION")
                        print("   Same image not recognized")
                else:
                    print("❌ Face recognition accuracy: NO FACES DETECTED")
                    
            except json.JSONDecodeError:
                print("❌ Face recognition accuracy: Invalid JSON response")
        else:
            print(f"❌ Face recognition accuracy: HTTP Error {response.status_code}")
        
        # Cleanup
        self.session.get(f"{BASE_URL}/delete_student/{unique_student_id}")
        return False
    
    def run_all_diagnostics(self):
        """Run all diagnostic tests"""
        print("🚀 CRITICAL ISSUES DIAGNOSTIC")
        print("=" * 60)
        
        if not self.login():
            print("❌ Cannot proceed without login")
            return False
        
        results = {
            'student_crud': self.test_student_crud_operations(),
            'photo_recognition_display': self.test_photo_recognition_display(),
            'face_recognition_accuracy': self.test_face_recognition_accuracy()
        }
        
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        passed = sum(results.values())
        total = len(results)
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All critical systems are working!")
        else:
            print("⚠️ Critical issues detected. See details above.")
        
        return passed == total

def main():
    """Main function"""
    diagnostic = CriticalIssuesDiagnostic()
    success = diagnostic.run_all_diagnostics()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
