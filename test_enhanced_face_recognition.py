#!/usr/bin/env python3
"""
Test script to verify enhanced face recognition accuracy
"""

import os
import sys
import time
import requests
import json
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

class EnhancedFaceRecognitionTester:
    def __init__(self):
        self.session = requests.Session()
        self.logged_in = False
        self.test_students = []
    
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
    
    def create_unique_test_image(self, student_id, variation=0):
        """Create a unique test image for a student"""
        # Create a more distinctive image for each student
        img = Image.new('RGB', (400, 400), color=(200 + variation * 10, 220 + variation * 5, 240))
        draw = ImageDraw.Draw(img)
        
        # Draw a unique face pattern for each student
        base_color = (150 + variation * 20, 180 + variation * 15, 200 + variation * 10)
        
        # Face outline
        draw.ellipse([80, 80, 320, 320], fill=base_color, outline='black', width=3)
        
        # Eyes (unique positions for each student)
        eye_offset = variation * 5
        draw.ellipse([130 + eye_offset, 150, 160 + eye_offset, 180], fill='black')
        draw.ellipse([240 - eye_offset, 150, 270 - eye_offset, 180], fill='black')
        
        # Nose (unique shape)
        nose_points = [(200, 190 + variation), (190 + variation, 220), (210 - variation, 220)]
        draw.polygon(nose_points, fill='brown')
        
        # Mouth (unique curve)
        mouth_y = 250 + variation * 2
        draw.arc([170, mouth_y, 230, mouth_y + 30], 0, 180, fill='red', width=3)
        
        # Add student ID as text
        draw.text((150, 340), f"ID: {student_id}", fill='black', anchor="mm")
        
        # Add some noise for realism
        for _ in range(50):
            x, y = np.random.randint(0, 400, 2)
            color = tuple(np.random.randint(0, 255, 3))
            draw.point((x, y), fill=color)
        
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        return img_buffer
    
    def create_test_students(self, count=5):
        """Create test students with unique images"""
        print(f"\n📝 CREATING {count} TEST STUDENTS")
        print("=" * 50)
        
        for i in range(count):
            student_id = f"ENHANCED{int(time.time())}{i:02d}"
            
            # Create unique image for this student
            test_image = self.create_unique_test_image(student_id, i)
            
            student_data = {
                'student_id': student_id,
                'name': f'Enhanced Test Student {i+1}',
                'email': f'enhanced.test.{student_id}@email.com',
                'department': 'Computer Science',
                'year': str((i % 4) + 1)
            }
            
            files = {'face_image': (f'enhanced_test_{i}.jpg', test_image, 'image/jpeg')}
            response = self.session.post(f"{BASE_URL}/add_student", data=student_data, files=files)
            
            if response.status_code == 200 and 'successfully' in response.text.lower():
                print(f"✅ Created student {i+1}: {student_id}")
                self.test_students.append({
                    'student_id': student_id,
                    'name': student_data['name'],
                    'variation': i
                })
            else:
                print(f"❌ Failed to create student {i+1}: {student_id}")
        
        print(f"\n✅ Created {len(self.test_students)} test students")
        return len(self.test_students) > 0
    
    def test_same_image_recognition(self):
        """Test recognition accuracy with identical images"""
        print(f"\n🎯 TESTING SAME IMAGE RECOGNITION ACCURACY")
        print("=" * 50)
        
        if not self.test_students:
            print("❌ No test students available")
            return False
        
        high_accuracy_count = 0
        total_tests = 0
        results = []
        
        for student in self.test_students:
            student_id = student['student_id']
            variation = student['variation']
            
            print(f"\n--- Testing Student: {student_id} ---")
            
            # Create the EXACT same image used during registration
            test_image = self.create_unique_test_image(student_id, variation)
            
            # Test recognition
            files = {'photo': (f'recognition_test_{student_id}.jpg', test_image, 'image/jpeg')}
            response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
            
            total_tests += 1
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    if result.get('success') and result.get('faces_recognized', 0) > 0:
                        recognized_student = result['data'][0]
                        confidence = recognized_student.get('confidence', 0)
                        confidence_percent = confidence * 100
                        
                        if recognized_student['student_id'] == student_id:
                            print(f"✅ CORRECT MATCH: {confidence_percent:.1f}% confidence")
                            
                            if confidence_percent >= 95:
                                print(f"🎉 HIGH ACCURACY: {confidence_percent:.1f}% (≥95%)")
                                high_accuracy_count += 1
                            elif confidence_percent >= 80:
                                print(f"✅ GOOD ACCURACY: {confidence_percent:.1f}% (≥80%)")
                            else:
                                print(f"⚠️ LOW ACCURACY: {confidence_percent:.1f}% (<80%)")
                            
                            results.append({
                                'student_id': student_id,
                                'correct_match': True,
                                'confidence': confidence_percent,
                                'high_accuracy': confidence_percent >= 95
                            })
                        else:
                            print(f"❌ WRONG MATCH: Got {recognized_student['student_id']} instead of {student_id}")
                            results.append({
                                'student_id': student_id,
                                'correct_match': False,
                                'confidence': confidence_percent,
                                'high_accuracy': False
                            })
                    else:
                        print(f"❌ NO RECOGNITION: No faces recognized")
                        results.append({
                            'student_id': student_id,
                            'correct_match': False,
                            'confidence': 0,
                            'high_accuracy': False
                        })
                        
                except json.JSONDecodeError:
                    print(f"❌ INVALID RESPONSE: Could not parse JSON")
                    results.append({
                        'student_id': student_id,
                        'correct_match': False,
                        'confidence': 0,
                        'high_accuracy': False
                    })
            else:
                print(f"❌ HTTP ERROR: {response.status_code}")
                results.append({
                    'student_id': student_id,
                    'correct_match': False,
                    'confidence': 0,
                    'high_accuracy': False
                })
        
        # Calculate statistics
        correct_matches = sum(1 for r in results if r['correct_match'])
        avg_confidence = sum(r['confidence'] for r in results if r['correct_match']) / max(correct_matches, 1)
        
        print(f"\n📊 RECOGNITION ACCURACY RESULTS")
        print("=" * 40)
        print(f"Total Tests: {total_tests}")
        print(f"Correct Matches: {correct_matches}/{total_tests} ({correct_matches/total_tests*100:.1f}%)")
        print(f"High Accuracy (≥95%): {high_accuracy_count}/{total_tests} ({high_accuracy_count/total_tests*100:.1f}%)")
        print(f"Average Confidence: {avg_confidence:.1f}%")
        
        # Success criteria
        success = (
            correct_matches / total_tests >= 0.95 and  # 95% correct matches
            high_accuracy_count / total_tests >= 0.8   # 80% high accuracy
        )
        
        if success:
            print(f"🎉 SUCCESS: Enhanced face recognition meets accuracy requirements!")
        else:
            print(f"⚠️ NEEDS IMPROVEMENT: Recognition accuracy below target")
        
        return success
    
    def test_real_time_recognition(self):
        """Test real-time recognition endpoint"""
        print(f"\n📹 TESTING REAL-TIME RECOGNITION")
        print("=" * 50)
        
        if not self.test_students:
            print("❌ No test students available")
            return False
        
        # Test with first student
        student = self.test_students[0]
        student_id = student['student_id']
        variation = student['variation']
        
        print(f"Testing real-time recognition with: {student_id}")
        
        # Create test image
        test_image = self.create_unique_test_image(student_id, variation)
        
        # Convert to base64 for real-time endpoint
        import base64
        image_data = base64.b64encode(test_image.getvalue()).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_data}"
        
        # Test real-time recognition
        response = self.session.post(f"{BASE_URL}/recognize_realtime", 
                                   json={'image': image_data_url})
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                if result.get('success'):
                    faces_detected = result.get('faces_detected', 0)
                    faces_recognized = result.get('faces_recognized', 0)
                    
                    print(f"✅ Real-time recognition working")
                    print(f"   Faces detected: {faces_detected}")
                    print(f"   Faces recognized: {faces_recognized}")
                    
                    if faces_recognized > 0:
                        recognized = result['data'][0]
                        if recognized['student_id'] == student_id:
                            confidence = recognized['confidence'] * 100
                            print(f"   ✅ Correct match: {confidence:.1f}% confidence")
                            return True
                        else:
                            print(f"   ❌ Wrong match: {recognized['student_id']}")
                    else:
                        print(f"   ⚠️ No faces recognized")
                else:
                    print(f"❌ Real-time recognition failed: {result.get('message', 'Unknown error')}")
                    
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
        
        return False
    
    def cleanup_test_students(self):
        """Clean up test students"""
        print(f"\n🧹 CLEANING UP TEST STUDENTS")
        print("=" * 50)
        
        cleaned_count = 0
        for student in self.test_students:
            student_id = student['student_id']
            response = self.session.get(f"{BASE_URL}/delete_student/{student_id}")
            
            if response.status_code in [200, 302]:
                print(f"✅ Deleted: {student_id}")
                cleaned_count += 1
            else:
                print(f"❌ Failed to delete: {student_id}")
        
        print(f"✅ Cleaned up {cleaned_count}/{len(self.test_students)} test students")
    
    def run_comprehensive_test(self):
        """Run comprehensive enhanced face recognition test"""
        print("🚀 ENHANCED FACE RECOGNITION COMPREHENSIVE TEST")
        print("=" * 60)
        
        if not self.login():
            print("❌ Cannot proceed without login")
            return False
        
        # Create test students
        if not self.create_test_students(5):
            print("❌ Failed to create test students")
            return False
        
        # Wait for processing
        print("\n⏳ Waiting for face embedding processing...")
        time.sleep(3)
        
        # Test same image recognition accuracy
        accuracy_success = self.test_same_image_recognition()
        
        # Test real-time recognition
        realtime_success = self.test_real_time_recognition()
        
        # Cleanup
        self.cleanup_test_students()
        
        # Final results
        print(f"\n" + "=" * 60)
        print("📊 FINAL TEST RESULTS")
        print("=" * 60)
        print(f"Same Image Recognition: {'✅ PASS' if accuracy_success else '❌ FAIL'}")
        print(f"Real-time Recognition: {'✅ PASS' if realtime_success else '❌ FAIL'}")
        
        overall_success = accuracy_success and realtime_success
        
        if overall_success:
            print(f"🎉 OVERALL: ✅ ENHANCED FACE RECOGNITION SYSTEM WORKING!")
        else:
            print(f"⚠️ OVERALL: ❌ SOME ISSUES DETECTED")
        
        return overall_success

def main():
    """Main function"""
    tester = EnhancedFaceRecognitionTester()
    success = tester.run_comprehensive_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
