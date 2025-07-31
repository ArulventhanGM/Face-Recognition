#!/usr/bin/env python3
"""
Debug the attendance route variables
"""

import sys
sys.path.append('.')
from utils.data_manager import get_data_manager
from datetime import datetime, timedelta
from collections import Counter

def debug_attendance_variables():
    """Debug all variables that should be passed to attendance template"""
    
    print("🔍 Debugging Attendance Route Variables...")
    print("=" * 60)
    
    # Get data manager
    data_manager = get_data_manager()
    
    # Get all data
    all_students = data_manager.get_all_students()
    all_attendance = data_manager.get_all_attendance()
    
    print(f"📊 Raw Data:")
    print(f"   Students: {len(all_students)}")
    print(f"   Attendance: {len(all_attendance)}")
    
    # Test filtering (simulate no filters)
    date_from = None
    date_to = None
    student_id = None
    department = None
    method = None
    
    # Apply filters
    filtered_attendance = all_attendance.copy()
    
    if date_from:
        filtered_attendance = [att for att in filtered_attendance if att.get('date', '') >= date_from]
    if date_to:
        filtered_attendance = [att for att in filtered_attendance if att.get('date', '') <= date_to]
    if student_id:
        filtered_attendance = [att for att in filtered_attendance if att.get('student_id', '') == student_id]
    if department:
        filtered_attendance = [att for att in filtered_attendance if att.get('department', '') == department]
    if method:
        filtered_attendance = [att for att in filtered_attendance if att.get('method', '') == method]
    
    print(f"\\n📋 Filtered Data:")
    print(f"   Filtered Attendance: {len(filtered_attendance)}")
    if filtered_attendance:
        print(f"   Sample Record: {filtered_attendance[0]}")
    
    # Calculate statistics
    today = datetime.now().strftime('%Y-%m-%d')
    this_week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime('%Y-%m-%d')
    this_month_start = datetime.now().strftime('%Y-%m-01')
    
    today_attendance = [att for att in filtered_attendance if att.get('date', '') == today]
    week_attendance = [att for att in filtered_attendance if att.get('date', '') >= this_week_start]
    month_attendance = [att for att in filtered_attendance if att.get('date', '') >= this_month_start]
    
    stats = {
        'total_today': len(today_attendance),
        'total_week': len(week_attendance),
        'total_month': len(month_attendance),
        'daily_average': len(month_attendance) // max(1, datetime.now().day)
    }
    
    print(f"\\n📈 Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Get departments
    departments = list(set([student.get('department', 'Unknown') for student in all_students]))
    departments.sort()
    
    print(f"\\n🏢 Departments:")
    print(f"   Count: {len(departments)}")
    print(f"   List: {departments}")
    
    # Attendance insights
    last_7_days = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    attendance_by_date = {}
    for date in last_7_days:
        count = len([att for att in all_attendance if att['date'] == date])
        attendance_by_date[date] = count
    
    print(f"\\n📅 Attendance by Date (Last 7 days):")
    for date, count in attendance_by_date.items():
        print(f"   {date}: {count}")
    
    # Department-wise attendance
    attendance_by_department = Counter([att.get('department', 'Unknown') for att in all_attendance])
    
    print(f"\\n🏢 Attendance by Department:")
    for dept, count in attendance_by_department.items():
        print(f"   {dept}: {count}")
    
    # Check what template variables would be
    template_vars = {
        'attendance_records': filtered_attendance,
        'stats': stats,
        'departments': departments,
        'attendance_by_date': attendance_by_date,
        'attendance_by_department': attendance_by_department
    }
    
    print(f"\\n🎯 Template Variables Summary:")
    for var_name, var_value in template_vars.items():
        if isinstance(var_value, list):
            print(f"   {var_name}: List with {len(var_value)} items")
        elif isinstance(var_value, dict):
            print(f"   {var_name}: Dict with {len(var_value)} keys")
        else:
            print(f"   {var_name}: {type(var_value).__name__}")
    
    return template_vars

if __name__ == "__main__":
    debug_attendance_variables()
