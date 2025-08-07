#!/usr/bin/env python3
"""
Enhanced Photo Recognition Test Script
Tests the complete photo upload and recognition workflow
"""

import requests
import json
import os
import sys
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

class PhotoRecognitionTester:
    def __init__(self):
        self.session = requests.Session()
        self.logged_in = False
    
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
            
            if response.status_code == 200 and 'dashboard' in response.url:
                print("✅ Login successful")
                self.logged_in = True
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def create_test_image(self, has_face=True, num_faces=1):
        """Create a test image with or without faces"""
        # Create a simple test image
        img = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        if has_face:
            for i in range(num_faces):
                # Draw simple face-like shapes
                x_offset = i * 120 + 50
                y_offset = 50
                
                # Face circle
                draw.ellipse([x_offset, y_offset, x_offset + 100, y_offset + 100], 
                           fill='peachpuff', outline='black')
                
                # Eyes
                draw.ellipse([x_offset + 20, y_offset + 30, x_offset + 30, y_offset + 40], 
                           fill='black')
                draw.ellipse([x_offset + 70, y_offset + 30, x_offset + 80, y_offset + 40], 
                           fill='black')
                
                # Mouth
                draw.arc([x_offset + 30, y_offset + 60, x_offset + 70, y_offset + 80], 
                        0, 180, fill='black')
        else:
            # Draw some non-face shapes
            draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
            draw.ellipse([200, 100, 300, 200], fill='green', outline='black')
        
        # Save to BytesIO
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        return img_buffer
    
    def test_no_faces_scenario(self):
        """Test scenario where no faces are detected"""
        print("\n🧪 Testing: No faces detected scenario")
        
        try:
            # Create image without faces
            test_image = self.create_test_image(has_face=False)
            
            files = {'photo': ('no_faces.jpg', test_image, 'image/jpeg')}
            response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                
                if result.get('success') and result.get('faces_detected') == 0:
                    print("✅ No faces scenario handled correctly")
                    return True
                else:
                    print("⚠️ Unexpected response for no faces scenario")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in no faces test: {e}")
            return False
    
    def test_single_face_scenario(self):
        """Test scenario with single face"""
        print("\n🧪 Testing: Single face scenario")
        
        try:
            # Create image with one face
            test_image = self.create_test_image(has_face=True, num_faces=1)
            
            files = {'photo': ('single_face.jpg', test_image, 'image/jpeg')}
            response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                
                if result.get('success') and result.get('faces_detected') >= 1:
                    print("✅ Single face scenario handled correctly")
                    return True
                else:
                    print("⚠️ Unexpected response for single face scenario")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in single face test: {e}")
            return False
    
    def test_multiple_faces_scenario(self):
        """Test scenario with multiple faces"""
        print("\n🧪 Testing: Multiple faces scenario")
        
        try:
            # Create image with multiple faces
            test_image = self.create_test_image(has_face=True, num_faces=3)
            
            files = {'photo': ('multiple_faces.jpg', test_image, 'image/jpeg')}
            response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                
                if result.get('success') and result.get('faces_detected') >= 2:
                    print("✅ Multiple faces scenario handled correctly")
                    return True
                else:
                    print("⚠️ Unexpected response for multiple faces scenario")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in multiple faces test: {e}")
            return False
    
    def test_invalid_file_scenario(self):
        """Test scenario with invalid file"""
        print("\n🧪 Testing: Invalid file scenario")
        
        try:
            # Create a text file instead of image
            invalid_file = BytesIO(b"This is not an image file")
            
            files = {'photo': ('invalid.txt', invalid_file, 'text/plain')}
            response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                
                if not result.get('success'):
                    print("✅ Invalid file scenario handled correctly")
                    return True
                else:
                    print("⚠️ Invalid file was accepted unexpectedly")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error in invalid file test: {e}")
            return False
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print("🚀 Starting Enhanced Photo Recognition Tests")
        print("=" * 60)
        
        if not self.login():
            print("❌ Cannot proceed without login")
            return False
        
        tests = [
            self.test_no_faces_scenario,
            self.test_single_face_scenario,
            self.test_multiple_faces_scenario,
            self.test_invalid_file_scenario
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        print("\n" + "=" * 60)
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All enhanced photo recognition tests PASSED!")
            return True
        else:
            print("⚠️ Some tests failed. Please check the implementation.")
            return False

def main():
    """Main function"""
    tester = PhotoRecognitionTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
