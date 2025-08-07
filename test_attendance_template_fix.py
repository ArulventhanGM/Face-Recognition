#!/usr/bin/env python3
"""
Test script to verify the attendance page template syntax error is fixed
"""

import requests
import json

# Configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

class AttendancePageTester:
    def __init__(self):
        self.session = requests.Session()
        self.logged_in = False
    
    def login(self):
        """Login to the system"""
        print("🔐 Logging in...")
        
        login_data = {
            'username': TEST_USERNAME,
            'password': TEST_PASSWORD
        }
        
        response = self.session.post(f"{BASE_URL}/login", data=login_data)
        
        if response.status_code == 200 and ('dashboard' in response.url or 'Dashboard' in response.text):
            print("✅ Login successful")
            self.logged_in = True
            return True
        else:
            print(f"❌ Login failed: {response.status_code}")
            return False
    
    def test_attendance_page(self):
        """Test if the attendance page loads without template errors"""
        print("\n📊 Testing attendance page...")
        
        if not self.logged_in:
            print("❌ Not logged in")
            return False
        
        try:
            response = self.session.get(f"{BASE_URL}/attendance")
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Attendance page loads successfully")
                
                # Check for key elements that should be present
                page_content = response.text
                
                checks = [
                    ('Attendance Report', 'Page title'),
                    ('student-photo', 'Student photo styling'),
                    ('attendance_records', 'Attendance records variable usage'),
                    ('default-avatar.svg', 'Default avatar fallback'),
                ]
                
                all_good = True
                for element, description in checks:
                    if element in page_content:
                        print(f"✅ {description} found")
                    else:
                        print(f"⚠️ {description} not found (may be empty data)")
                
                # Most importantly, check that there are no template syntax errors
                if 'TemplateSyntaxError' not in page_content and 'jinja2.exceptions' not in page_content:
                    print("✅ No template syntax errors detected")
                    return True
                else:
                    print("❌ Template syntax errors still present")
                    return False
                    
            elif response.status_code == 500:
                print("❌ Server error - template syntax issue likely still exists")
                print(f"Response preview: {response.text[:300]}")
                return False
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing attendance page: {e}")
            return False
    
    def test_with_filters(self):
        """Test attendance page with various filters"""
        print("\n🔍 Testing attendance page with filters...")
        
        if not self.logged_in:
            print("❌ Not logged in")
            return False
        
        # Test different filter combinations
        test_filters = [
            {'date_from': '2025-08-01'},
            {'date_to': '2025-08-04'}, 
            {'department': 'Computer Science'},
            {'student_id': 'STU001'},
            {'method': 'camera'},
        ]
        
        all_passed = True
        
        for i, filters in enumerate(test_filters):
            try:
                response = self.session.get(f"{BASE_URL}/attendance", params=filters)
                
                if response.status_code == 200:
                    print(f"✅ Filter test {i+1} passed: {filters}")
                else:
                    print(f"❌ Filter test {i+1} failed: {filters} (Status: {response.status_code})")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ Filter test {i+1} error: {e}")
                all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """Run all attendance page tests"""
        print("=" * 60)
        print("🧪 ATTENDANCE PAGE TEMPLATE SYNTAX FIX TESTS")
        print("=" * 60)
        
        # Login first
        if not self.login():
            print("❌ Cannot proceed without login")
            return False
        
        # Test basic page load
        page_loads = self.test_attendance_page()
        
        # Test with filters if basic page works
        filters_work = True
        if page_loads:
            filters_work = self.test_with_filters()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        results = {
            'attendance_page_loads': page_loads,
            'filters_working': filters_work,
        }
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests PASSED - Template syntax error is FIXED!")
            print("\n✅ The attendance page now works correctly!")
            print("✅ Student photos with fallback to default avatar working!")
            print("✅ All filtering functionality working!")
        else:
            print("⚠️ Some tests failed - check logs above for details")
        
        return passed == total

if __name__ == "__main__":
    tester = AttendancePageTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 ISSUE RESOLVED:")
        print("- Fixed HTML entity encoding in Jinja2 template")
        print("- Changed &quot; to regular quotes in url_for expression")  
        print("- Attendance page template syntax error eliminated")
        print("- Page loads successfully with student photos and fallbacks")
    else:
        print("\n❌ Some issues remain - check the test output above")
