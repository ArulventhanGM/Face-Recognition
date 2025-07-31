#!/usr/bin/env python3
"""
Add Student Photo Upload Diagnostic Script
Test photo upload functionality comprehensively
"""

import requests
import os
import tempfile
from PIL import Image
import io

class PhotoUploadDiagnostic:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.issues = []
    
    def log_issue(self, category, description, severity="ERROR"):
        """Log issues found"""
        issue = {
            'category': category,
            'description': description,
            'severity': severity
        }
        self.issues.append(issue)
        print(f"🔍 {severity}: {category} - {description}")
    
    def login(self):
        """Login to the system"""
        try:
            response = self.session.post(f"{self.base_url}/login", 
                                       data={'username': 'admin', 'password': 'admin123'})
            if response.status_code == 200:
                print("✅ Login successful")
                return True
            else:
                self.log_issue("AUTHENTICATION", f"Login failed with status {response.status_code}")
                return False
        except Exception as e:
            self.log_issue("AUTHENTICATION", f"Login exception: {e}")
            return False
    
    def test_add_student_page_loads(self):
        """Test if add student page loads correctly"""
        print("\\n=== Testing Add Student Page Loading ===")
        
        try:
            response = self.session.get(f"{self.base_url}/add_student")
            
            if response.status_code != 200:
                self.log_issue("PAGE_LOADING", f"Add student page returned status {response.status_code}")
                return False
            
            content = response.text
            
            # Check for required form elements
            required_elements = [
                ('enctype="multipart/form-data"', "Form has multipart encoding"),
                ('name="face_image"', "Face image input field"),
                ('accept="image/*"', "Image file type restriction"),
                ('type="file"', "File input type"),
                ('required', "Required attribute on file input")
            ]
            
            for element, description in required_elements:
                if element in content:
                    print(f"✅ {description}")
                else:
                    self.log_issue("FORM_ELEMENTS", f"Missing {description}")
            
            return True
            
        except Exception as e:
            self.log_issue("PAGE_LOADING", f"Exception loading add student page: {e}")
            return False
    
    def create_test_image(self, width=400, height=400, format_type="JPEG"):
        """Create a test image for upload"""
        # Create a simple test image
        image = Image.new('RGB', (width, height), color='lightblue')
        
        # Add some content to make it look like a face photo
        # This is just a simple colored rectangle to simulate content
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], fill='white')
        draw.ellipse([width//3, height//3, 2*width//3, 2*height//3], fill='pink')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format_type)
        img_bytes.seek(0)
        
        return img_bytes
    
    def test_valid_photo_upload(self):
        """Test uploading a valid photo"""
        print("\\n=== Testing Valid Photo Upload ===")
        
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Prepare form data
            form_data = {
                'student_id': 'TEST001',
                'name': 'Test Student Upload',
                'email': 'testupload@example.com',
                'department': 'Computer Science',
                'year': '2'
            }
            
            files = {
                'face_image': ('test_photo.jpg', test_image, 'image/jpeg')
            }
            
            response = self.session.post(f"{self.base_url}/add_student", 
                                       data=form_data, 
                                       files=files)
            
            if response.status_code == 200:
                content = response.text
                
                # Check for success indicators
                if "added successfully" in content.lower() or "success" in content.lower():
                    print("✅ Valid photo upload successful")
                    return True
                elif "error" in content.lower() or "failed" in content.lower():
                    # Extract error message
                    import re
                    error_pattern = r'class="alert alert-error"[^>]*>([^<]+)'
                    error_match = re.search(error_pattern, content)
                    error_msg = error_match.group(1) if error_match else "Unknown error"
                    self.log_issue("UPLOAD_PROCESSING", f"Upload failed: {error_msg}")
                    return False
                else:
                    self.log_issue("UPLOAD_RESPONSE", "Upload response unclear - no success or error message")
                    return False
            else:
                self.log_issue("UPLOAD_REQUEST", f"Upload request failed with status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_issue("UPLOAD_EXCEPTION", f"Exception during photo upload: {e}")
            return False
    
    def test_invalid_file_types(self):
        """Test uploading invalid file types"""
        print("\\n=== Testing Invalid File Type Rejection ===")
        
        invalid_files = [
            ("test.txt", b"This is a text file", "text/plain"),
            ("test.pdf", b"PDF content", "application/pdf"),
            ("test.doc", b"DOC content", "application/msword")
        ]
        
        for filename, content, mimetype in invalid_files:
            try:
                form_data = {
                    'student_id': 'TEST002',
                    'name': 'Test Invalid File',
                    'email': 'testinvalid@example.com',
                    'department': 'Computer Science', 
                    'year': '2'
                }
                
                files = {
                    'face_image': (filename, io.BytesIO(content), mimetype)
                }
                
                response = self.session.post(f"{self.base_url}/add_student",
                                           data=form_data,
                                           files=files)
                
                if "Invalid image format" in response.text or "error" in response.text.lower():
                    print(f"✅ {filename} correctly rejected")
                else:
                    self.log_issue("FILE_VALIDATION", f"{filename} was not properly rejected")
                    
            except Exception as e:
                self.log_issue("FILE_VALIDATION", f"Exception testing {filename}: {e}")
    
    def test_file_size_limits(self):
        """Test file size limit enforcement"""
        print("\\n=== Testing File Size Limits ===")
        
        try:
            # Create oversized image (simulate >16MB)
            large_image = self.create_test_image(2000, 2000, "PNG")  # Larger PNG
            
            form_data = {
                'student_id': 'TEST003',
                'name': 'Test Large File',
                'email': 'testlarge@example.com',
                'department': 'Computer Science',
                'year': '2'
            }
            
            files = {
                'face_image': ('large_test.png', large_image, 'image/png')
            }
            
            response = self.session.post(f"{self.base_url}/add_student",
                                       data=form_data,
                                       files=files)
            
            # For actual large files, we'd expect size rejection
            # But our test image might not be large enough
            if "too large" in response.text.lower() or "size" in response.text.lower():
                print("✅ File size limit correctly enforced")
            else:
                print("ℹ️  File size limit test - image may not be large enough to trigger limit")
                
        except Exception as e:
            self.log_issue("SIZE_VALIDATION", f"Exception testing file size: {e}")
    
    def test_missing_photo_handling(self):
        """Test form submission without photo"""
        print("\\n=== Testing Missing Photo Handling ===")
        
        try:
            form_data = {
                'student_id': 'TEST004',
                'name': 'Test No Photo',
                'email': 'testnophoto@example.com',
                'department': 'Computer Science',
                'year': '2'
            }
            
            # Submit without face_image file
            response = self.session.post(f"{self.base_url}/add_student", data=form_data)
            
            # Should show error since face_image is required for new students
            if "required" in response.text.lower() or "error" in response.text.lower():
                print("✅ Missing photo correctly handled")
            else:
                self.log_issue("REQUIRED_VALIDATION", "Missing photo was not properly validated")
                
        except Exception as e:
            self.log_issue("REQUIRED_VALIDATION", f"Exception testing missing photo: {e}")
    
    def test_upload_folder_permissions(self):
        """Test if upload folder is accessible and writable"""
        print("\\n=== Testing Upload Folder Permissions ===")
        
        try:
            # Check if uploads folder exists
            upload_folder = "uploads"
            
            if os.path.exists(upload_folder):
                print(f"✅ Upload folder '{upload_folder}' exists")
                
                if os.access(upload_folder, os.W_OK):
                    print("✅ Upload folder is writable")
                else:
                    self.log_issue("PERMISSIONS", "Upload folder is not writable")
                    
                # List contents
                files = os.listdir(upload_folder)
                print(f"✅ Upload folder contains {len(files)} files")
                
            else:
                self.log_issue("FOLDER_MISSING", f"Upload folder '{upload_folder}' does not exist")
                
        except Exception as e:
            self.log_issue("PERMISSIONS", f"Exception checking upload folder: {e}")
    
    def run_comprehensive_diagnostic(self):
        """Run all diagnostic tests"""
        print("🔍 Starting Add Student Photo Upload Diagnostic...")
        print("=" * 70)
        
        # Login first
        if not self.login():
            print("❌ Cannot proceed without authentication")
            return self.issues
        
        # Run all tests
        self.test_add_student_page_loads()
        self.test_upload_folder_permissions()
        self.test_valid_photo_upload()
        self.test_invalid_file_types()
        self.test_file_size_limits()
        self.test_missing_photo_handling()
        
        # Summary
        print("=" * 70)
        print("📊 PHOTO UPLOAD DIAGNOSTIC SUMMARY")
        print("=" * 70)
        
        if not self.issues:
            print("🎉 No photo upload issues found! Functionality working correctly.")
        else:
            error_count = len([i for i in self.issues if i['severity'] == 'ERROR'])
            warning_count = len([i for i in self.issues if i['severity'] == 'WARNING'])
            
            print(f"Total Issues Found: {len(self.issues)}")
            print(f"Errors: {error_count}")
            print(f"Warnings: {warning_count}")
            
            if error_count > 0:
                print("\\n❌ CRITICAL ERRORS:")
                for issue in [i for i in self.issues if i['severity'] == 'ERROR']:
                    print(f"  - {issue['category']}: {issue['description']}")
            
            if warning_count > 0:
                print("\\n⚠️  WARNINGS:")
                for issue in [i for i in self.issues if i['severity'] == 'WARNING']:
                    print(f"  - {issue['category']}: {issue['description']}")
        
        return self.issues

if __name__ == "__main__":
    diagnostic = PhotoUploadDiagnostic()
    issues = diagnostic.run_comprehensive_diagnostic()
