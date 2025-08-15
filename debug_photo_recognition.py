#!/usr/bin/env python3
"""
Debug script to test photo recognition functionality
"""
import requests
import json
import os

def test_photo_recognition():
    """Test the photo recognition endpoint"""
    
    # Test with a sample image (you can replace this with an actual image path)
    test_image_path = "test_images/sample.jpg"  # Replace with actual image
    
    # If no test image exists, create a simple one for testing
    if not os.path.exists(test_image_path):
        print("No test image found. Please upload an image through the web interface to test.")
        return
    
    url = "http://localhost:5000/recognize_photo"
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'photo': f}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response JSON:")
                print(json.dumps(result, indent=2))
                
                if result.get('success'):
                    print(f"\n✅ Recognition successful!")
                    print(f"Faces detected: {result.get('faces_detected', 0)}")
                    print(f"Faces recognized: {result.get('faces_recognized', 0)}")
                    print(f"Recognition data: {result.get('data', [])}")
                else:
                    print(f"\n❌ Recognition failed: {result.get('message')}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response: {e}")
                print(f"Raw response: {response.text}")
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure Flask app is running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_photo_recognition()
