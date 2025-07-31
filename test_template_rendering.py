#!/usr/bin/env python3
"""
Test the actual template rendering with debug output
"""

import requests
import re

def test_template_rendering():
    """Test actual template variable rendering"""
    
    session = requests.Session()
    
    # Login
    response = session.post("http://localhost:5000/login", 
                           data={'username': 'admin', 'password': 'admin123'})
    
    if response.status_code != 200:
        print("❌ Login failed")
        return
    
    # Get attendance page
    response = session.get("http://localhost:5000/attendance")
    content = response.text
    
    print("🔍 Testing Template Variable Rendering...")
    print("=" * 60)
    
    # Test 1: Check if statistics are rendered
    print("\\n📊 Statistics Rendering:")
    stat_pattern = r'<span class="stat-number">(\d+)</span>'
    stat_matches = re.findall(stat_pattern, content)
    
    if stat_matches:
        print(f"✅ Found statistics: {stat_matches}")
        
        # Check specific statistics
        stat_labels = ["Present Today", "This Week", "This Month", "Daily Average"]
        for i, label in enumerate(stat_labels):
            if label in content and i < len(stat_matches):
                print(f"   ✅ {label}: {stat_matches[i]}")
            else:
                print(f"   ❌ {label}: Not found or no value")
    else:
        print("❌ No statistics found")
    
    # Test 2: Check attendance records count
    print("\\n📋 Attendance Records:")
    
    # Look for record count badge
    count_pattern = r'<span class="badge badge-primary">(\d+) records</span>'
    count_match = re.search(count_pattern, content)
    
    if count_match:
        record_count = count_match.group(1)
        print(f"✅ Found record count badge: {record_count} records")
    else:
        print("❌ Record count badge not found")
    
    # Look for actual table rows
    row_pattern = r'<tr class="attendance-row"'
    row_matches = re.findall(row_pattern, content)
    
    if row_matches:
        print(f"✅ Found {len(row_matches)} attendance table rows")
    else:
        print("❌ No attendance table rows found")
    
    # Test 3: Check departments in filter
    print("\\n🏢 Department Filter:")
    
    # Look for department options
    dept_pattern = r'<option value="([^"]+)"[^>]*>([^<]+)</option>'
    dept_matches = re.findall(dept_pattern, content)
    
    if dept_matches:
        print(f"✅ Found {len(dept_matches)} department options:")
        for value, text in dept_matches[:5]:  # Show first 5
            print(f"   - {text} ({value})")
    else:
        print("❌ No department options found")
    
    # Test 4: Check for JavaScript data (charts)
    print("\\n📈 Chart Data:")
    
    # Look for JavaScript chart data
    if "attendance_by_date" in content:
        print("✅ Found attendance_by_date data for charts")
    else:
        print("❌ attendance_by_date data not found")
    
    if "attendance_by_department" in content:
        print("✅ Found attendance_by_department data for charts")
    else:
        print("❌ attendance_by_department data not found")
    
    # Test 5: Check for template error indicators
    print("\\n⚠️  Error Detection:")
    
    error_indicators = [
        ("{{ ", "Unrendered template variable"),
        ("undefined", "Undefined variable"),
        ("TemplateRuntimeError", "Template runtime error"),
        ("jinja2", "Jinja2 error")
    ]
    
    for indicator, description in error_indicators:
        if indicator in content:
            print(f"❌ Found {description}: {indicator}")
        else:
            print(f"✅ No {description}")
    
    # Test 6: Check if template extends properly
    print("\\n🔧 Template Structure:")
    
    if "<title>Attendance - Face Recognition System</title>" in content:
        print("✅ Template title is rendered correctly")
    else:
        print("❌ Template title not found")
    
    if "Attendance Management" in content:
        print("✅ Main heading is rendered")
    else:
        print("❌ Main heading not found")
    
    if "Filter Attendance Records" in content:
        print("✅ Filter section is rendered")
    else:
        print("❌ Filter section not found")
    
    # Save a snippet for manual inspection
    print(f"\\n📄 Content Length: {len(content)} characters")
    
    # Find the stats section specifically
    stats_start = content.find('<div class="stats-grid">')
    if stats_start != -1:
        stats_end = content.find('</div>', stats_start) + 6
        stats_snippet = content[stats_start:stats_end]
        print(f"\\n📊 Stats Section HTML:")
        print(stats_snippet[:500] + "..." if len(stats_snippet) > 500 else stats_snippet)

if __name__ == "__main__":
    test_template_rendering()
