#!/usr/bin/env python3
"""
Simple Photo Upload Test
"""

import requests
import os

def test_photo_upload():
    """Test photo upload functionality"""
    
    session = requests.Session()
    
    print("🔍 Testing Photo Upload Functionality...")
    print("=" * 50)
    
    # Login
    try:
        response = session.post("http://localhost:5000/login", 
                               data={'username': 'admin', 'password': 'admin123'})
        if response.status_code == 200:
            print("✅ Login successful")
        else:
            print("❌ Login failed")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    # Test 1: Check add student page
    print("\\n1. Testing Add Student Page:")
    try:
        response = session.get("http://localhost:5000/add_student")
        if response.status_code == 200:
            print("✅ Add student page loads")
            
            content = response.text
            
            # Check form elements
            checks = [
                ('enctype="multipart/form-data"', "Form multipart encoding"),
                ('name="face_image"', "Face image input"),
                ('accept="image/*"', "Image file filter"),
                ('type="file"', "File input type")
            ]
            
            for check, desc in checks:
                if check in content:
                    print(f"✅ {desc} present")
                else:
                    print(f"❌ {desc} missing")
        else:
            print(f"❌ Page load failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Page test error: {e}")
    
    # Test 2: Check upload folder
    print("\\n2. Testing Upload Folder:")
    upload_folder = "uploads"
    try:
        if os.path.exists(upload_folder):
            print(f"✅ Upload folder '{upload_folder}' exists")
            
            if os.access(upload_folder, os.W_OK):
                print("✅ Upload folder is writable")
            else:
                print("❌ Upload folder is not writable")
            
            files = os.listdir(upload_folder)
            print(f"✅ Upload folder contains {len(files)} files")
        else:
            print(f"❌ Upload folder '{upload_folder}' does not exist")
    except Exception as e:
        print(f"❌ Upload folder test error: {e}")
    
    # Test 3: Try submitting form without photo
    print("\\n3. Testing Form Submission Without Photo:")
    try:
        form_data = {
            'student_id': 'TESTNO001',
            'name': 'Test No Photo',
            'email': 'testnophoto@example.com',
            'department': 'Computer Science',
            'year': '2'
        }
        
        response = session.post("http://localhost:5000/add_student", data=form_data)
        
        if response.status_code == 200:
            content = response.text
            if "error" in content.lower() or "required" in content.lower():
                print("✅ Missing photo properly validated")
            else:
                print("❌ Missing photo validation may be missing")
        else:
            print(f"❌ Form submission failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Form test error: {e}")
    
    # Test 4: Test with a simple text file as image (should fail)
    print("\\n4. Testing Invalid File Type:")
    try:
        form_data = {
            'student_id': 'TESTINV001',
            'name': 'Test Invalid File',
            'email': 'testinvalid@example.com',
            'department': 'Computer Science',
            'year': '2'
        }
        
        files = {
            'face_image': ('test.txt', b'This is not an image', 'text/plain')
        }
        
        response = session.post("http://localhost:5000/add_student", 
                               data=form_data, files=files)
        
        if response.status_code == 200:
            content = response.text
            if "invalid" in content.lower() or "format" in content.lower():
                print("✅ Invalid file type properly rejected")
            else:
                print("❌ Invalid file type validation may be missing")
        else:
            print(f"❌ Invalid file test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid file test error: {e}")

if __name__ == "__main__":
    test_photo_upload()
