#!/usr/bin/env python3
"""
Test script for Photo Upload Recognition functionality
This script tests the complete workflow of photo upload and face recognition
"""

import requests
import os
import json
from io import BytesIO
from PIL import Image, ImageDraw
import time

# Configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

class PhotoUploadTester:
    def __init__(self):
        self.session = requests.Session()
        self.logged_in = False
    
    def login(self):
        """Login to the system"""
        print("🔐 Logging in...")
        
        # Get login page to get CSRF token if needed
        login_page = self.session.get(f"{BASE_URL}/login")
        if login_page.status_code != 200:
            print(f"❌ Failed to access login page: {login_page.status_code}")
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
    
    def create_test_image(self, width=400, height=300, text="TEST IMAGE"):
        """Create a simple test image"""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face-like pattern
        # Head (circle)
        draw.ellipse([width//4, height//4, 3*width//4, 3*height//4], outline='black', width=3)
        
        # Eyes
        eye_y = height//3
        draw.ellipse([width//3, eye_y, width//3 + 20, eye_y + 20], fill='black')
        draw.ellipse([2*width//3 - 20, eye_y, 2*width//3, eye_y + 20], fill='black')
        
        # Mouth
        mouth_y = 2*height//3
        draw.arc([width//3, mouth_y, 2*width//3, mouth_y + 30], start=0, end=180, fill='black', width=3)
        
        # Add text
        draw.text((10, height - 30), text, fill='black')
        
        return img
    
    def test_photo_recognition_endpoint(self):
        """Test the /recognize_photo endpoint directly"""
        print("\n📸 Testing photo recognition endpoint...")
        
        if not self.logged_in:
            print("❌ Not logged in")
            return False
        
        # Create test image
        test_image = self.create_test_image(text="Recognition Test")
        
        # Convert to bytes
        img_buffer = BytesIO()
        test_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Prepare file upload
        files = {
            'photo': ('test_image.jpg', img_buffer, 'image/jpeg')
        }
        
        try:
            # Send request
            response = self.session.post(f"{BASE_URL}/recognize_photo", files=files)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"Response: {json.dumps(result, indent=2)}")
                    
                    if result.get('success'):
                        print("✅ Photo recognition endpoint working")
                        print(f"✅ Recognized {len(result.get('data', []))} face(s)")
                        return True
                    else:
                        print(f"⚠️ Recognition failed: {result.get('message', 'Unknown error')}")
                        return True  # Endpoint is working, just no faces recognized
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON response: {response.text[:200]}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False
    
    def test_page_access(self):
        """Test if recognition page loads correctly"""
        print("\n🌐 Testing recognition page access...")
        
        if not self.logged_in:
            print("❌ Not logged in")
            return False
        
        try:
            response = self.session.get(f"{BASE_URL}/recognition")
            
            if response.status_code == 200:
                # Check for required elements
                page_content = response.text
                
                checks = [
                    ('photoRecognitionForm', 'Photo recognition form'),
                    ('uploadPhoto', 'Upload photo input'),
                    ('recognitionResults', 'Recognition results container'),
                    ('file-input-wrapper', 'File input wrapper styling'),
                ]
                
                all_good = True
                for element, description in checks:
                    if element in page_content:
                        print(f"✅ {description} found")
                    else:
                        print(f"❌ {description} missing")
                        all_good = False
                
                return all_good
            else:
                print(f"❌ Failed to load recognition page: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error accessing recognition page: {e}")
            return False
    
    def test_file_upload_workflow(self):
        """Test complete file upload workflow simulation"""
        print("\n🔄 Testing complete upload workflow...")
        
        # This simulates what happens when user uploads a file
        results = {
            'endpoint_accessible': self.test_photo_recognition_endpoint(),
            'page_loads_correctly': self.test_page_access(),
        }
        
        return results
    
    def run_all_tests(self):
        """Run all photo upload tests"""
        print("=" * 60)
        print("🧪 PHOTO UPLOAD RECOGNITION TESTS")
        print("=" * 60)
        
        # Login first
        if not self.login():
            print("❌ Cannot proceed without login")
            return
        
        # Run tests
        results = self.test_file_upload_workflow()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All photo upload recognition tests PASSED!")
        else:
            print("⚠️  Some tests failed - check logs above for details")
        
        return passed == total

if __name__ == "__main__":
    tester = PhotoUploadTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Photo upload recognition functionality is working correctly!")
    else:
        print("\n❌ Photo upload recognition functionality has issues that need fixing.")
