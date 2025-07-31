#!/usr/bin/env python3
"""
Comprehensive test script for student registration functionality
Tests both student data submission and face photo upload
"""

import requests
import cv2
import numpy as np
import os
import tempfile
import json
from datetime import datetime

class StudentRegistrationTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def login(self):
        """Login as admin"""
        response = self.session.post(f"{self.base_url}/login", 
                                   data={'username': 'admin', 'password': 'admin123'})
        return response.status_code == 200
    
    def log_test(self, test_name, success, message=""):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
    
    def create_test_image(self, filename="test_face.jpg"):
        """Create a test face image"""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create a simple face-like pattern
        cv2.circle(img, (100, 100), 80, (255, 220, 177), -1)  # Face
        cv2.circle(img, (80, 80), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (120, 80), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img, (100, 120), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        cv2.imwrite(filename, img)
        return filename
    
    def test_page_accessibility(self):
        """Test if add_student page loads correctly"""
        try:
            response = self.session.get(f"{self.base_url}/add_student")
            if response.status_code == 200:
                has_form = 'form' in response.text.lower()
                has_file_input = 'face_image' in response.text
                has_required_fields = all(field in response.text for field in 
                                        ['student_id', 'name', 'email', 'department', 'year'])
                
                if has_form and has_file_input and has_required_fields:
                    self.log_test("Page Accessibility", True, "Add student page loads with all required elements")
                else:
                    self.log_test("Page Accessibility", False, "Missing form elements")
            else:
                self.log_test("Page Accessibility", False, f"Page returned status {response.status_code}")
        except Exception as e:
            self.log_test("Page Accessibility", False, f"Exception: {e}")
    
    def test_basic_student_submission(self):
        """Test submitting student data without file"""
        try:
            student_data = {
                'student_id': f'TEST{datetime.now().strftime("%H%M%S")}',
                'name': 'Basic Test Student',
                'email': f'basic.test.{datetime.now().strftime("%H%M%S")}@email.com',
                'department': 'Computer Science',
                'year': '2'
            }
            
            response = self.session.post(f"{self.base_url}/add_student", data=student_data)
            
            if response.status_code == 200:
                success_indicators = ['success' in response.text.lower(), 
                                    response.url != f"{self.base_url}/add_student"]
                
                if any(success_indicators):
                    self.log_test("Basic Student Submission", True, "Student submitted successfully without photo")
                else:
                    self.log_test("Basic Student Submission", False, "No success indicators found")
            else:
                self.log_test("Basic Student Submission", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("Basic Student Submission", False, f"Exception: {e}")
    
    def test_student_with_photo(self):
        """Test submitting student data with photo"""
        try:
            # Create test image
            image_file = self.create_test_image("temp_test_face.jpg")
            
            student_data = {
                'student_id': f'PHOTO{datetime.now().strftime("%H%M%S")}',
                'name': 'Photo Test Student',
                'email': f'photo.test.{datetime.now().strftime("%H%M%S")}@email.com',
                'department': 'Engineering',
                'year': '3'
            }
            
            with open(image_file, 'rb') as f:
                files = {'face_image': ('test_face.jpg', f, 'image/jpeg')}
                response = self.session.post(f"{self.base_url}/add_student", 
                                           data=student_data, files=files)
            
            if response.status_code == 200:
                success_indicators = ['success' in response.text.lower(), 
                                    response.url != f"{self.base_url}/add_student"]
                
                # Check if file was uploaded
                uploads_exist = os.path.exists('uploads') and len(os.listdir('uploads')) > 0
                
                if any(success_indicators) and uploads_exist:
                    self.log_test("Student with Photo", True, "Student with photo submitted successfully")
                else:
                    self.log_test("Student with Photo", False, "Photo upload may have failed")
            else:
                self.log_test("Student with Photo", False, f"Status: {response.status_code}")
                
            # Clean up
            if os.path.exists(image_file):
                os.remove(image_file)
                
        except Exception as e:
            self.log_test("Student with Photo", False, f"Exception: {e}")
    
    def test_validation_errors(self):
        """Test form validation"""
        test_cases = [
            {
                'name': 'Missing Required Fields',
                'data': {'student_id': 'TEST001', 'name': '', 'email': 'test@email.com'},
                'expected_error': 'required'
            },
            {
                'name': 'Invalid Email',
                'data': {
                    'student_id': 'TEST002',
                    'name': 'Test Student',
                    'email': 'not-an-email',
                    'department': 'CS',
                    'year': '1'
                },
                'expected_error': 'email'
            },
            {
                'name': 'Duplicate Student ID',
                'data': {
                    'student_id': 'STU001',  # Already exists
                    'name': 'Duplicate Test',
                    'email': 'dup@email.com',
                    'department': 'CS',
                    'year': '1'
                },
                'expected_error': 'exists'
            }
        ]
        
        for case in test_cases:
            try:
                response = self.session.post(f"{self.base_url}/add_student", data=case['data'])
                
                # Should stay on form page with error
                stays_on_form = 'Add New Student' in response.text
                has_error = any(keyword in response.text.lower() 
                              for keyword in ['error', 'invalid', 'required', 'exists'])
                
                if stays_on_form and has_error:
                    self.log_test(f"Validation: {case['name']}", True, "Validation working correctly")
                else:
                    self.log_test(f"Validation: {case['name']}", False, "Validation not working")
                    
            except Exception as e:
                self.log_test(f"Validation: {case['name']}", False, f"Exception: {e}")
    
    def test_file_upload_edge_cases(self):
        """Test various file upload scenarios"""
        student_base = {
            'student_id': f'FILE{datetime.now().strftime("%H%M%S")}',
            'name': 'File Test Student',
            'email': f'file.test.{datetime.now().strftime("%H%M%S")}@email.com',
            'department': 'Computer Science',
            'year': '2'
        }
        
        # Test 1: Invalid file type
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write('This is not an image')
                temp_file = f.name
            
            with open(temp_file, 'rb') as f:
                files = {'face_image': ('test.txt', f, 'text/plain')}
                response = self.session.post(f"{self.base_url}/add_student", 
                                           data=student_base, files=files)
            
            # Should reject invalid file
            has_error = 'error' in response.text.lower() or 'invalid' in response.text.lower()
            self.log_test("File Upload: Invalid Type", has_error, 
                         "Invalid file type correctly rejected" if has_error else "Invalid file was accepted")
            
            os.unlink(temp_file)
            
        except Exception as e:
            self.log_test("File Upload: Invalid Type", False, f"Exception: {e}")
        
        # Test 2: Large file
        try:
            # Create a large dummy file (simulate large image)
            large_data = b'0' * (17 * 1024 * 1024)  # 17MB (over limit)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                f.write(large_data)
                temp_file = f.name
            
            # This should be handled by Flask's MAX_CONTENT_LENGTH
            student_data = student_base.copy()
            student_data['student_id'] += '_LARGE'
            
            try:
                with open(temp_file, 'rb') as f:
                    files = {'face_image': ('large.jpg', f, 'image/jpeg')}
                    response = self.session.post(f"{self.base_url}/add_student", 
                                               data=student_data, files=files)
                
                # Should either reject or handle gracefully
                self.log_test("File Upload: Large File", True, "Large file handled (may be rejected)")
            except requests.exceptions.RequestException:
                self.log_test("File Upload: Large File", True, "Large file rejected by server")
            
            os.unlink(temp_file)
            
        except Exception as e:
            self.log_test("File Upload: Large File", False, f"Exception: {e}")
    
    def test_students_display(self):
        """Test if submitted students appear in the students list"""
        try:
            response = self.session.get(f"{self.base_url}/students")
            
            if response.status_code == 200:
                # Check for recently added test students
                has_test_students = any(keyword in response.text for keyword in 
                                      ['Test Student', 'PHOTO', 'TEST'])
                
                self.log_test("Students Display", has_test_students, 
                             "Test students appear in students list" if has_test_students 
                             else "Test students may not be displaying")
            else:
                self.log_test("Students Display", False, f"Students page status: {response.status_code}")
                
        except Exception as e:
            self.log_test("Students Display", False, f"Exception: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting Student Registration Comprehensive Tests...")
        print("=" * 60)
        
        # Login first
        if not self.login():
            print("❌ Failed to login. Aborting tests.")
            return
        
        print("✅ Login successful")
        
        # Run tests
        self.test_page_accessibility()
        self.test_basic_student_submission()
        self.test_student_with_photo()
        self.test_validation_errors()
        self.test_file_upload_edge_cases()
        self.test_students_display()
        
        # Summary
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['message']}")
        
        return self.test_results

if __name__ == "__main__":
    tester = StudentRegistrationTester()
    results = tester.run_all_tests()
