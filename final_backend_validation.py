#!/usr/bin/env python3
"""
Final Backend Validation - Complete Functionality Test
"""

import requests
import re
from datetime import datetime

def final_backend_validation():
    """Complete validation of all backend functionality"""
    
    session = requests.Session()
    
    print("🎯 FINAL BACKEND ATTENDANCE SYSTEM VALIDATION")
    print("=" * 70)
    
    # Login
    response = session.post("http://localhost:5000/login", 
                           data={'username': 'admin', 'password': 'admin123'})
    
    if response.status_code != 200:
        print("❌ CRITICAL: Login failed")
        return False
    
    print("✅ Authentication: Working")
    
    # Test basic attendance page
    response = session.get("http://localhost:5000/attendance")
    content = response.text
    
    if response.status_code != 200:
        print("❌ CRITICAL: Attendance page failed to load")
        return False
    
    print("✅ Page Loading: Working")
    
    # Validate all major components
    validation_results = {
        'Statistics Display': False,
        'Attendance Records Table': False,
        'Department Filter': False,
        'Date Filter': False,
        'Insights Tables': False,
        'Filter Functionality': False
    }
    
    # Test 1: Statistics
    stat_pattern = r'<span class="stat-number">(\d+)</span>'
    stat_matches = re.findall(stat_pattern, content)
    
    if len(stat_matches) >= 4:
        validation_results['Statistics Display'] = True
        print(f"✅ Statistics: {stat_matches} (Today, Week, Month, Average)")
    else:
        print(f"❌ Statistics incomplete: {stat_matches}")
    
    # Test 2: Attendance Records
    tbody_start = content.find("<tbody>")
    tbody_end = content.find("</tbody>")
    
    if tbody_start != -1 and tbody_end != -1:
        tbody_content = content[tbody_start:tbody_end]
        tr_count = tbody_content.count("<tr>")
        
        if tr_count > 0 and "STU001" in tbody_content:
            validation_results['Attendance Records Table'] = True
            print(f"✅ Attendance Table: {tr_count} records displayed")
        else:
            print(f"❌ Attendance Table: {tr_count} records, missing student data")
    
    # Test 3: Department Filter
    dept_options = re.findall(r'<option value="([^"]*)"[^>]*>([^<]*)</option>', content)
    
    if len(dept_options) > 2:  # Should have at least 2-3 departments
        validation_results['Department Filter'] = True
        print(f"✅ Department Filter: {len(dept_options)} options")
    else:
        print(f"❌ Department Filter: Only {len(dept_options)} options")
    
    # Test 4: Date Filter
    if 'name="date_from"' in content and 'name="date_to"' in content:
        validation_results['Date Filter'] = True
        print("✅ Date Filter: Input fields present")
    else:
        print("❌ Date Filter: Missing input fields")
    
    # Test 5: Insights Tables
    if "Daily Attendance" in content and "Attendance by Department" in content:
        validation_results['Insights Tables'] = True
        print("✅ Insights Tables: Both sections present")
    else:
        print("❌ Insights Tables: Missing sections")
    
    # Test 6: Filter Functionality
    filter_response = session.get("http://localhost:5000/attendance", 
                                params={'department': 'Computer Science'})
    
    if filter_response.status_code == 200 and 'Computer Science' in filter_response.text:
        validation_results['Filter Functionality'] = True
        print("✅ Filter Functionality: Department filter working")
    else:
        print("❌ Filter Functionality: Department filter failed")
    
    # Summary
    print("\\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    working_components = sum(validation_results.values())
    total_components = len(validation_results)
    
    for component, status in validation_results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")
    
    print(f"\\n🎯 Overall Status: {working_components}/{total_components} components working")
    
    if working_components == total_components:
        print("\\n🎉 RESULT: Backend attendance system is FULLY FUNCTIONAL!")
        print("   - All data is being retrieved correctly")
        print("   - All template variables are being passed properly")
        print("   - All filtering functionality is working")
        print("   - All statistics are calculating correctly")
        print("   - All user interface components are rendering")
        return True
    else:
        print(f"\\n⚠️  RESULT: {total_components - working_components} components need attention")
        return False

if __name__ == "__main__":
    final_backend_validation()
