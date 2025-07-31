#!/usr/bin/env python3
"""
Fixed test for template rendering focusing on actual table structure
"""

import requests
import re

def test_attendance_table():
    """Test attendance table rendering specifically"""
    
    session = requests.Session()
    
    # Login
    response = session.post("http://localhost:5000/login", 
                           data={'username': 'admin', 'password': 'admin123'})
    
    # Get attendance page
    response = session.get("http://localhost:5000/attendance")
    content = response.text
    
    print("🔍 Testing Attendance Table Rendering...")
    print("=" * 60)
    
    # Check if we have attendance records
    print("\\n📊 Attendance Records Check:")
    
    # Look for "13 records" badge
    if "13 records" in content:
        print("✅ Found record count: 13 records")
    else:
        print("❌ Record count not found")
    
    # Check if the table exists
    if '<div class="table-container">' in content:
        print("✅ Table container found")
    else:
        print("❌ Table container not found")
    
    if '<table class="table">' in content:
        print("✅ Table element found")
    else:
        print("❌ Table element not found")
    
    # Check for table headers
    headers = ["Date", "Time", "Student ID", "Student Name", "Department", "Year", "Method", "Confidence", "Status"]
    for header in headers:
        if f"<th>{header}</th>" in content:
            print(f"✅ Header '{header}' found")
        else:
            print(f"❌ Header '{header}' not found")
    
    # Look for tbody
    if "<tbody>" in content:
        print("✅ Table body found")
    else:
        print("❌ Table body not found")
    
    # Extract the table body content specifically
    tbody_start = content.find("<tbody>")
    tbody_end = content.find("</tbody>")
    
    if tbody_start != -1 and tbody_end != -1:
        tbody_content = content[tbody_start:tbody_end + 8]
        print(f"\\n📋 Table Body Content ({len(tbody_content)} chars):")
        
        # Count <tr> tags in tbody (excluding header)
        tr_count = tbody_content.count("<tr>")
        print(f"   Table rows found: {tr_count}")
        
        if tr_count == 0:
            print("❌ No table rows found in tbody")
            print("\\n🔍 Tbody content preview:")
            print(tbody_content[:500])
        else:
            print(f"✅ Found {tr_count} table rows")
            
            # Check for student data
            if "STU001" in tbody_content:
                print("✅ Found student ID STU001 in table")
            else:
                print("❌ Student ID STU001 not found in table")
                
            if "John Doe" in tbody_content:
                print("✅ Found student name 'John Doe' in table") 
            else:
                print("❌ Student name 'John Doe' not found in table")
    
    # Check the conditional rendering
    print("\\n🔧 Conditional Rendering Check:")
    
    # Look for the conditional block
    if "{% if attendance_records %}" in content:
        print("❌ Found raw template syntax - template not rendering properly!")
    else:
        print("✅ No raw template syntax found")
    
    # Check for "No records found" message
    if "No attendance records found" in content:
        print("❌ Found 'No records found' message - records not being passed")
    else:
        print("✅ No 'no records found' message")
    
    # Check attendance_records specifically
    print("\\n📋 Debugging Attendance Records:")
    
    # Look for any mention of attendance records or loops
    for_pattern = r'{% for (\w+) in (\w+) %}'
    for_matches = re.findall(for_pattern, content)
    
    if for_matches:
        print("❌ Found unrendered for loops:")
        for var, collection in for_matches:
            print(f"   - {{% for {var} in {collection} %}}")
    else:
        print("✅ No unrendered for loops found")
    
    # Look for specific data that should be in the table
    sample_data = ["2025-07-30", "09:30:00", "Present", "camera"]
    found_data = []
    
    for data in sample_data:
        if data in content:
            found_data.append(data)
            print(f"✅ Found sample data: {data}")
        else:
            print(f"❌ Sample data not found: {data}")
    
    print(f"\\n📊 Summary: Found {len(found_data)}/{len(sample_data)} sample data items")

if __name__ == "__main__":
    test_attendance_table()
