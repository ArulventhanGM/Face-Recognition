#!/usr/bin/env python3
"""
Test actual photo upload with a real image file
"""

import requests
import os
import tempfile

def test_real_photo_upload():
    """Test uploading an actual image file"""
    
    session = requests.Session()
    
    print("🔍 Testing Real Photo Upload...")
    print("=" * 40)
    
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
    
    # Create a simple test image using basic image creation
    print("\\n📷 Creating test image...")
    
    try:
        # Create a minimal PNG file (1x1 pixel)
        # PNG file header for a 1x1 transparent pixel
        png_data = b'\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f\\x15\\xc4\\x89\\x00\\x00\\x00\\rIDATx\\x9cc```\\x00\\x01\\x00\\x00\\x05\\x00\\x01\\r\\n-\\xdb\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82'
        
        # Test with real image upload
        form_data = {
            'student_id': 'TESTREAL001',
            'name': 'Test Real Upload',
            'email': 'testreal@example.com',
            'department': 'Computer Science',
            'year': '2'
        }
        
        files = {
            'face_image': ('test_photo.png', png_data, 'image/png')
        }
        
        print("📤 Uploading test student with photo...")
        response = session.post("http://localhost:5000/add_student", 
                               data=form_data, files=files)
        
        if response.status_code == 200:
            content = response.text
            
            # Check for success
            if "added successfully" in content.lower() or "success" in content.lower():
                print("✅ Photo upload successful!")
                print("   Student added with face photo")
                
                # Check if file was saved
                upload_files = os.listdir("uploads")
                png_files = [f for f in upload_files if f.endswith('.png')]
                if png_files:
                    print(f"✅ Photo file saved: {png_files[-1]}")
                else:
                    print("❌ Photo file not found in uploads folder")
                    
            elif "error" in content.lower():
                print("❌ Upload failed with error")
                # Try to extract error message
                start = content.lower().find("error")
                if start != -1:
                    error_section = content[start:start+200]
                    print(f"   Error details: {error_section}")
            else:
                print("❌ Upload response unclear")
                print(f"   Response length: {len(content)} chars")
        else:
            print(f"❌ Upload request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Real upload test error: {e}")
    
    print("\\n📊 Upload folder status:")
    try:
        files = os.listdir("uploads")
        print(f"   Total files: {len(files)}")
        recent_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join("uploads", x)), reverse=True)[:3]
        print(f"   Recent files: {recent_files}")
    except Exception as e:
        print(f"   Error checking uploads: {e}")

if __name__ == "__main__":
    test_real_photo_upload()
