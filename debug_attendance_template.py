#!/usr/bin/env python3
"""
Attendance Template Debugging Script
Tests various aspects of the attendance.html template
"""

import requests
import json
from datetime import datetime, timedelta

class AttendanceTemplateDebugger:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.issues = []
    
    def login(self):
        """Login as admin"""
        response = self.session.post(f"{self.base_url}/login", 
                                   data={'username': 'admin', 'password': 'admin123'})
        return response.status_code == 200
    
    def log_issue(self, issue_type, description, severity="WARNING"):
        """Log a found issue"""
        issue = {
            'type': issue_type,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        self.issues.append(issue)
        print(f"🔍 {severity}: {issue_type} - {description}")
    
    def test_basic_rendering(self):
        """Test basic template rendering"""
        print("\\n=== Testing Basic Template Rendering ===")
        
        response = self.session.get(f"{self.base_url}/attendance")
        
        if response.status_code != 200:
            self.log_issue("HTTP_ERROR", f"Attendance page returned status {response.status_code}", "ERROR")
            return False
        
        content = response.text
        
        # Check for Jinja2 errors
        if "TemplateError" in content or "UndefinedError" in content:
            self.log_issue("TEMPLATE_ERROR", "Jinja2 template errors found", "ERROR")
        
        # Check for required sections
        required_sections = [
            ("stats-grid", "Statistics section"),
            ("Filter Attendance Records", "Filter form"),
            ("Attendance Records", "Records table"),
            ("confidence-bar", "Confidence indicators"),
        ]
        
        for selector, name in required_sections:
            if selector not in content:
                self.log_issue("MISSING_SECTION", f"Missing {name}", "WARNING")
        
        return True
    
    def test_data_rendering(self):
        """Test how template handles data"""
        print("\\n=== Testing Data Rendering ===")
        
        response = self.session.get(f"{self.base_url}/attendance")
        content = response.text
        
        # Check for data display
        if "STU001" in content or "John Doe" in content:
            print("✅ Student data is rendering correctly")
        else:
            self.log_issue("DATA_MISSING", "Student data not appearing in template", "WARNING")
        
        # Check confidence display
        if "confidence-fill" in content:
            print("✅ Confidence indicators are rendering")
        else:
            self.log_issue("CONFIDENCE_MISSING", "Confidence indicators not rendering", "WARNING")
        
        # Check for method badges
        if "badge-success" in content and "badge-warning" in content:
            print("✅ Method badges are rendering")
        else:
            self.log_issue("BADGES_MISSING", "Method badges not rendering properly", "WARNING")
    
    def test_confidence_calculation(self):
        """Test confidence percentage calculation"""
        print("\\n=== Testing Confidence Calculation ===")
        
        response = self.session.get(f"{self.base_url}/attendance")
        content = response.text
        
        # Look for confidence percentages
        confidence_indicators = ["85%", "90%", "95%"]  # Based on test data
        found_percentages = []
        
        for indicator in confidence_indicators:
            if indicator in content:
                found_percentages.append(indicator)
        
        if found_percentages:
            print(f"✅ Confidence percentages found: {found_percentages}")
        else:
            self.log_issue("CONFIDENCE_CALC", "Confidence percentages not calculating correctly", "ERROR")
    
    def test_filter_functionality(self):
        """Test filter form functionality"""
        print("\\n=== Testing Filter Functionality ===")
        
        # Test date filtering
        response = self.session.get(f"{self.base_url}/attendance", params={
            'date_from': '2025-07-30',
            'date_to': '2025-07-30'
        })
        
        if response.status_code == 200:
            print("✅ Date filtering works")
        else:
            self.log_issue("FILTER_ERROR", "Date filtering not working", "ERROR")
        
        # Test department filtering
        response = self.session.get(f"{self.base_url}/attendance", params={
            'department': 'Computer Science'
        })
        
        if response.status_code == 200:
            content = response.text
            if 'selected' in content:  # Check if filter is preserved
                print("✅ Department filtering works")
            else:
                self.log_issue("FILTER_PRESERVE", "Filter state not preserved", "WARNING")
        else:
            self.log_issue("FILTER_ERROR", "Department filtering not working", "ERROR")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\\n=== Testing Edge Cases ===")
        
        # Test with invalid date range
        response = self.session.get(f"{self.base_url}/attendance", params={
            'date_from': '2025-12-31',
            'date_to': '2025-01-01'  # Invalid range
        })
        
        if response.status_code == 200:
            print("✅ Invalid date range handled gracefully")
        else:
            self.log_issue("EDGE_CASE", "Invalid date range causes errors", "WARNING")
        
        # Test with non-existent student ID
        response = self.session.get(f"{self.base_url}/attendance", params={
            'student_id': 'NONEXISTENT999'
        })
        
        if response.status_code == 200:
            content = response.text
            if "No Attendance Records Found" in content:
                print("✅ Empty results handled correctly")
            else:
                self.log_issue("EMPTY_RESULTS", "Empty results not handled properly", "WARNING")
        else:
            self.log_issue("EDGE_CASE", "Non-existent student ID causes errors", "WARNING")
    
    def test_responsive_design(self):
        """Test responsive design elements"""
        print("\\n=== Testing Responsive Design ===")
        
        response = self.session.get(f"{self.base_url}/attendance")
        content = response.text
        
        # Check for responsive classes
        responsive_classes = ["grid", "d-flex", "gap-", "mt-", "ml-"]
        found_classes = []
        
        for cls in responsive_classes:
            if cls in content:
                found_classes.append(cls)
        
        if len(found_classes) >= 3:
            print(f"✅ Responsive classes found: {found_classes}")
        else:
            self.log_issue("RESPONSIVE", "Insufficient responsive design classes", "WARNING")
        
        # Check for mobile-friendly elements
        if "table-container" in content:
            print("✅ Table container for mobile scrolling present")
        else:
            self.log_issue("MOBILE", "Table container missing for mobile compatibility", "WARNING")
    
    def test_performance_indicators(self):
        """Test for potential performance issues"""
        print("\\n=== Testing Performance Indicators ===")
        
        response = self.session.get(f"{self.base_url}/attendance")
        
        # Check response time
        if hasattr(response, 'elapsed'):
            response_time = response.elapsed.total_seconds()
            if response_time > 2.0:
                self.log_issue("PERFORMANCE", f"Slow response time: {response_time:.2f}s", "WARNING")
            else:
                print(f"✅ Good response time: {response_time:.2f}s")
        
        # Check content size
        content_size = len(response.content)
        if content_size > 500000:  # 500KB
            self.log_issue("PERFORMANCE", f"Large page size: {content_size} bytes", "WARNING")
        else:
            print(f"✅ Reasonable page size: {content_size} bytes")
    
    def run_all_tests(self):
        """Run all debugging tests"""
        print("🔍 Starting Attendance Template Debugging...")
        print("=" * 60)
        
        if not self.login():
            print("❌ Failed to login. Cannot proceed with tests.")
            return
        
        print("✅ Login successful")
        
        # Run all tests
        self.test_basic_rendering()
        self.test_data_rendering()
        self.test_confidence_calculation()
        self.test_filter_functionality()
        self.test_edge_cases()
        self.test_responsive_design()
        self.test_performance_indicators()
        
        # Summary
        print("=" * 60)
        print("📊 DEBUGGING SUMMARY")
        print("=" * 60)
        
        if not self.issues:
            print("🎉 No issues found! Template is working correctly.")
        else:
            error_count = len([i for i in self.issues if i['severity'] == 'ERROR'])
            warning_count = len([i for i in self.issues if i['severity'] == 'WARNING'])
            
            print(f"Total Issues: {len(self.issues)}")
            print(f"Errors: {error_count}")
            print(f"Warnings: {warning_count}")
            
            if error_count > 0:
                print("\\n❌ CRITICAL ERRORS:")
                for issue in [i for i in self.issues if i['severity'] == 'ERROR']:
                    print(f"  - {issue['type']}: {issue['description']}")
            
            if warning_count > 0:
                print("\\n⚠️  WARNINGS:")
                for issue in [i for i in self.issues if i['severity'] == 'WARNING']:
                    print(f"  - {issue['type']}: {issue['description']}")
        
        return self.issues

if __name__ == "__main__":
    debugger = AttendanceTemplateDebugger()
    issues = debugger.run_all_tests()
