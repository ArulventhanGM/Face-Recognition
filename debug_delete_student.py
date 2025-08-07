#!/usr/bin/env python3
"""
Test script to debug the delete_student route issue
"""

import requests
import json

# Configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

class DeleteStudentTester:
    def __init__(self):
        self.session = requests.Session()
        self.logged_in = False
    
    def login(self):
        """Login to the system"""
        print("🔐 Logging in...")
        
        # Get login page
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
        
        if response.status_code == 200 and ('dashboard' in response.url or 'Dashboard' in response.text):
            print("✅ Login successful")
            self.logged_in = True
            return True
        else:
            print(f"❌ Login failed: {response.status_code}")
            return False
    
    def test_students_page(self):
        """Test if the students page loads and has delete links"""
        print("\n📝 Testing students page...")
        
        if not self.logged_in:
            print("❌ Not logged in")
            return False
        
        try:
            response = self.session.get(f"{BASE_URL}/students")
            
            if response.status_code == 200:
                print("✅ Students page loads successfully")
                
                # Check for delete links
                if 'delete_student' in response.text:
                    print("✅ Delete links found in students page")
                    
                    # Extract student IDs from the page
                    import re
                    delete_links = re.findall(r'/delete_student/([^"\']+)', response.text)
                    if delete_links:
                        print(f"✅ Found {len(delete_links)} delete links")
                        print(f"   Sample student IDs: {delete_links[:3]}")
                        return delete_links[0] if delete_links else None
                    else:
                        print("⚠️ No student IDs found in delete links")
                        return None
                else:
                    print("❌ No delete links found in students page")
                    return None
            else:
                print(f"❌ Students page failed to load: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error accessing students page: {e}")
            return None
    
    def test_delete_route(self, student_id):
        """Test the delete route directly"""
        print(f"\n🗑️ Testing delete route for student: {student_id}")
        
        if not self.logged_in:
            print("❌ Not logged in")
            return False
        
        try:
            delete_url = f"{BASE_URL}/delete_student/{student_id}"
            print(f"Attempting to access: {delete_url}")
            
            response = self.session.get(delete_url)
            
            print(f"Status Code: {response.status_code}")
            print(f"Final URL: {response.url}")
            
            if response.status_code == 200:
                print("✅ Delete route accessible")
                if 'students' in response.url or 'Students' in response.text:
                    print("✅ Redirected back to students page")
                    return True
                else:
                    print("⚠️ Route accessible but didn't redirect to students page")
                    return False
            elif response.status_code == 404:
                print("❌ 404 Error - Route not found")
                return False
            elif response.status_code == 302:
                print("✅ Redirect response (normal for delete)")
                return True
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                print(f"Response text preview: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing delete route: {e}")
            return False
    
    def test_flask_routes(self):
        """Test which routes are available"""
        print("\n🛣️ Testing available routes...")
        
        try:
            # Try to access a route that should give us route information
            response = self.session.get(f"{BASE_URL}/nonexistent_route_test")
            
            # If we get a proper 404 page, the Flask app is working
            if response.status_code == 404:
                print("✅ Flask app is responding to requests")
                
                # Test some known routes
                test_routes = [
                    '/dashboard',
                    '/students', 
                    '/attendance',
                    '/recognition'
                ]
                
                for route in test_routes:
                    try:
                        resp = self.session.get(f"{BASE_URL}{route}")
                        status = "✅" if resp.status_code == 200 else "❌"
                        print(f"   {route}: {status} {resp.status_code}")
                    except:
                        print(f"   {route}: ❌ Error")
                        
                return True
            else:
                print(f"⚠️ Unexpected response for non-existent route: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Flask routes: {e}")
            return False
    
    def run_all_tests(self):
        """Run all delete student tests"""
        print("=" * 60)
        print("🧪 DELETE STUDENT ROUTE DEBUG TESTS")
        print("=" * 60)
        
        # Login first
        if not self.login():
            print("❌ Cannot proceed without login")
            return
        
        # Test Flask routes
        self.test_flask_routes()
        
        # Test students page
        sample_student_id = self.test_students_page()
        
        # Test delete route if we have a student ID
        if sample_student_id:
            self.test_delete_route(sample_student_id)
        else:
            print("\n⚠️ No student ID available to test delete route")
        
        print("\n" + "=" * 60)
        print("🏁 DEBUG TESTS COMPLETE")
        print("=" * 60)

if __name__ == "__main__":
    tester = DeleteStudentTester()
    tester.run_all_tests()
