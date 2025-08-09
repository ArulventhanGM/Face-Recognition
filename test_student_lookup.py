#!/usr/bin/env python3
"""
Simple test to verify student lookup after fix
"""

from utils.data_manager import get_data_manager

def test_student_lookup():
    """Test student lookup"""
    print("Testing student lookup after data type fix...")
    
    dm = get_data_manager()
    
    # Test getting all students
    students = dm.get_all_students()
    print(f"Total students: {len(students)}")
    
    for student in students:
        student_id = student['student_id']
        name = student['name']
        print(f"Student: {student_id} ({type(student_id)}) - {name}")
        
        # Test individual lookup
        found_student = dm.get_student(student_id)
        if found_student:
            print(f"  ✅ Lookup successful")
        else:
            print(f"  ❌ Lookup failed")

if __name__ == "__main__":
    test_student_lookup()
