#!/usr/bin/env python3
"""
Backend Attendance Functionality Diagnostic Script
Comprehensive testing of attendance page backend functionality
"""

import requests
import json
import traceback
from datetime import datetime, timedelta

class AttendanceBackendDiagnostic:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.issues = []
    
    def log_issue(self, category, description, severity="ERROR"):
        """Log backend issues"""
        issue = {
            'category': category,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        self.issues.append(issue)
        print(f"🔍 {severity}: {category} - {description}")
    
    def login(self):
        """Test login functionality"""
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
    
    def test_basic_attendance_endpoint(self):
        """Test basic attendance page loading"""
        print("\\n=== Testing Basic Attendance Endpoint ===")
        
        try:
            response = self.session.get(f"{self.base_url}/attendance")
            
            if response.status_code != 200:
                self.log_issue("ENDPOINT", f"Attendance page returned status {response.status_code}")
                return False
            
            print(f"✅ Attendance endpoint responds with 200")
            print(f"   Content length: {len(response.content)} bytes")
            
            # Check for basic template elements
            content = response.text
            required_elements = [
                ("attendance_records", "attendance records data"),
                ("stats", "statistics section"),
                ("departments", "department filter"),
                ("Filter Attendance", "filter form")
            ]
            
            for element, description in required_elements:
                if element in content:
                    print(f"✅ Contains {description}")
                else:
                    self.log_issue("TEMPLATE_DATA", f"Missing {description}")
            
            return True
            
        except Exception as e:
            self.log_issue("ENDPOINT", f"Exception accessing attendance page: {e}")
            return False
    
    def test_filter_functionality(self):
        """Test attendance filtering functionality"""
        print("\\n=== Testing Filter Functionality ===")
        
        filter_tests = [
            ("date_from", {"date_from": "2025-07-30"}, "Date from filter"),
            ("date_to", {"date_to": "2025-07-31"}, "Date to filter"), 
            ("student_id", {"student_id": "STU001"}, "Student ID filter"),
            ("department", {"department": "Computer Science"}, "Department filter"),
            ("method", {"method": "camera"}, "Method filter"),
            ("combined", {"date_from": "2025-07-30", "department": "Computer Science"}, "Combined filters")
        ]
        
        for test_name, params, description in filter_tests:
            try:
                response = self.session.get(f"{self.base_url}/attendance", params=params)
                
                if response.status_code == 200:
                    print(f"✅ {description} responds correctly")
                    
                    # Check if filter parameters are preserved in form
                    content = response.text
                    for key, value in params.items():
                        if f'value="{value}"' in content:
                            print(f"   ✅ {key} parameter preserved in form")
                        else:
                            self.log_issue("FILTER_PRESERVATION", f"{key} parameter not preserved")
                else:
                    self.log_issue("FILTER_ENDPOINT", f"{description} failed with status {response.status_code}")
                    
            except Exception as e:
                self.log_issue("FILTER_EXCEPTION", f"{description} exception: {e}")
    
    def test_template_variables(self):
        """Test if all required template variables are provided"""
        print("\\n=== Testing Template Variables ===")
        
        try:
            response = self.session.get(f"{self.base_url}/attendance")
            content = response.text
            
            # Check for template variable usage
            template_vars = [
                ("attendance_records", "{% for record in attendance_records %}"),
                ("stats.total_today", "{{ stats.total_today }}"),
                ("departments", "{% for dept in departments %}"),
                ("attendance_by_date", "attendance_by_date"),
                ("attendance_by_department", "attendance_by_department")
            ]
            
            for var_name, pattern in template_vars:
                if pattern in content or var_name.replace('.', '_') in content:
                    print(f"✅ Template variable {var_name} is being used")
                else:
                    self.log_issue("TEMPLATE_VARS", f"Template variable {var_name} not found or not used")
                    
        except Exception as e:
            self.log_issue("TEMPLATE_VARS", f"Exception checking template variables: {e}")
    
    def test_data_manager_methods(self):
        """Test data manager methods directly"""
        print("\\n=== Testing Data Manager Methods ===")
        
        try:
            # Import and test data manager
            import sys
            sys.path.append('.')
            from utils.data_manager import get_data_manager
            
            data_manager = get_data_manager()
            
            # Test get_all_students
            students = data_manager.get_all_students()
            print(f"✅ get_all_students() returned {len(students)} students")
            
            if len(students) > 0:
                print(f"   Sample student: {students[0]}")
            else:
                self.log_issue("DATA_MANAGER", "No students found in database")
            
            # Test get_all_attendance  
            attendance = data_manager.get_all_attendance()
            print(f"✅ get_all_attendance() returned {len(attendance)} records")
            
            if len(attendance) > 0:
                print(f"   Sample attendance: {attendance[0]}")
            else:
                self.log_issue("DATA_MANAGER", "No attendance records found in database")
                
        except Exception as e:
            self.log_issue("DATA_MANAGER", f"Exception testing data manager: {e}")
    
    def test_attendance_statistics(self):
        """Test statistics calculation"""
        print("\\n=== Testing Statistics Calculation ===")
        
        try:
            response = self.session.get(f"{self.base_url}/attendance")
            content = response.text
            
            # Look for statistics in the HTML
            stat_indicators = ["Present Today", "This Week", "This Month", "Daily Average"]
            
            for indicator in stat_indicators:
                if indicator in content:
                    print(f"✅ Statistic '{indicator}' found in page")
                else:
                    self.log_issue("STATISTICS", f"Statistic '{indicator}' not found")
                    
            # Look for actual numbers
            import re
            stat_numbers = re.findall(r'<span class="stat-number">(\d+)</span>', content)
            if stat_numbers:
                print(f"✅ Found {len(stat_numbers)} statistic values: {stat_numbers}")
            else:
                self.log_issue("STATISTICS", "No statistic values found in page")
                
        except Exception as e:
            self.log_issue("STATISTICS", f"Exception testing statistics: {e}")
    
    def test_insights_data(self):
        """Test attendance insights data"""
        print("\\n=== Testing Insights Data ===")
        
        try:
            response = self.session.get(f"{self.base_url}/attendance")
            content = response.text
            
            # Check for insights sections
            insights = [
                ("Daily Attendance", "Last 7 Days"),
                ("Attendance by Department", "chart-pie")
            ]
            
            for insight_name, indicator in insights:
                if indicator in content:
                    print(f"✅ {insight_name} section found")
                else:
                    self.log_issue("INSIGHTS", f"{insight_name} section not found")
                    
        except Exception as e:
            self.log_issue("INSIGHTS", f"Exception testing insights: {e}")
    
    def test_error_handling(self):
        """Test error handling for edge cases"""
        print("\\n=== Testing Error Handling ===")
        
        edge_cases = [
            ("invalid_date_range", {"date_from": "2025-12-31", "date_to": "2025-01-01"}),
            ("non_existent_student", {"student_id": "NONEXISTENT999"}),
            ("invalid_department", {"department": "InvalidDepartment"}),
            ("empty_filters", {"date_from": "", "date_to": "", "student_id": ""})
        ]
        
        for test_name, params in edge_cases:
            try:
                response = self.session.get(f"{self.base_url}/attendance", params=params)
                
                if response.status_code == 200:
                    print(f"✅ {test_name} handled gracefully")
                else:
                    self.log_issue("ERROR_HANDLING", f"{test_name} caused error {response.status_code}")
                    
            except Exception as e:
                self.log_issue("ERROR_HANDLING", f"{test_name} exception: {e}")
    
    def run_comprehensive_diagnostic(self):
        """Run all diagnostic tests"""
        print("🔍 Starting Comprehensive Backend Attendance Diagnostic...")
        print("=" * 70)
        
        # Login first
        if not self.login():
            print("❌ Cannot proceed without authentication")
            return self.issues
        
        # Run all tests
        self.test_basic_attendance_endpoint()
        self.test_filter_functionality() 
        self.test_template_variables()
        self.test_data_manager_methods()
        self.test_attendance_statistics()
        self.test_insights_data()
        self.test_error_handling()
        
        # Summary
        print("=" * 70)
        print("📊 BACKEND DIAGNOSTIC SUMMARY")
        print("=" * 70)
        
        if not self.issues:
            print("🎉 No backend issues found! All functionality working correctly.")
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
    diagnostic = AttendanceBackendDiagnostic()
    issues = diagnostic.run_comprehensive_diagnostic()
