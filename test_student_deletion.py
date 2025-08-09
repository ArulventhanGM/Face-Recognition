#!/usr/bin/env python3
"""
Test script to verify student deletion functionality
"""

import os
import pickle
from utils.data_manager import get_data_manager

def test_deletion_workflow():
    """Test the complete deletion workflow"""
    print("🗑️ TESTING STUDENT DELETION WORKFLOW")
    print("=" * 50)
    
    dm = get_data_manager()
    
    # Get current students
    students = dm.get_all_students()
    print(f"Initial student count: {len(students)}")
    
    # Test deletion for each problematic student
    problematic_students = ['123456789', '987654321', '98765432134']
    
    for student_id in problematic_students:
        print(f"\n--- Testing deletion for Student ID: {student_id} ---")
        
        # Step 1: Verify student exists
        student = dm.get_student(student_id)
        if not student:
            print(f"  ❌ Student {student_id} not found - skipping")
            continue
        
        print(f"  ✅ Student found: {student['name']}")
        
        # Step 2: Check dependencies before deletion
        print("  🔍 Pre-deletion checks:")
        
        # Check face embedding
        has_embedding = student_id in dm.face_embeddings
        print(f"    - Face embedding: {'EXISTS' if has_embedding else 'None'}")
        
        # Check photo file
        photo_path = student.get('photo_path', '')
        has_photo = photo_path and photo_path != 'nan' and os.path.exists(photo_path)
        print(f"    - Photo file: {'EXISTS' if has_photo else 'None'}")
        if has_photo:
            print(f"      Path: {photo_path}")
        
        # Step 3: Perform deletion
        print("  🗑️ Attempting deletion...")
        success = dm.delete_student(student_id)
        
        if success:
            print("  ✅ Deletion successful")
            
            # Step 4: Verify deletion
            print("  🔍 Post-deletion verification:")
            
            # Check student no longer exists
            deleted_student = dm.get_student(student_id)
            if deleted_student is None:
                print("    ✅ Student removed from database")
            else:
                print("    ❌ Student still exists in database")
            
            # Check face embedding removed
            if has_embedding:
                if student_id not in dm.face_embeddings:
                    print("    ✅ Face embedding removed")
                else:
                    print("    ❌ Face embedding still exists")
            
            # Check photo file removed
            if has_photo:
                if not os.path.exists(photo_path):
                    print("    ✅ Photo file removed")
                else:
                    print("    ❌ Photo file still exists")
        else:
            print("  ❌ Deletion failed")
    
    # Final verification
    print(f"\n📊 FINAL VERIFICATION")
    print("=" * 30)
    
    remaining_students = dm.get_all_students()
    print(f"Remaining student count: {len(remaining_students)}")
    
    if len(remaining_students) == 0:
        print("✅ All problematic students successfully deleted")
    else:
        print("⚠️ Some students remain:")
        for student in remaining_students:
            print(f"  - {student['student_id']}: {student['name']}")

def backup_data():
    """Create backup before testing"""
    print("📋 Creating backup before testing...")
    
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Backup students.csv
    if os.path.exists('data/students.csv'):
        backup_path = f'data/students_backup_deletion_test_{timestamp}.csv'
        shutil.copy2('data/students.csv', backup_path)
        print(f"✅ Students CSV backed up to: {backup_path}")
    
    # Backup face embeddings
    if os.path.exists('data/face_embeddings.pkl'):
        backup_path = f'data/face_embeddings_backup_deletion_test_{timestamp}.pkl'
        shutil.copy2('data/face_embeddings.pkl', backup_path)
        print(f"✅ Face embeddings backed up to: {backup_path}")

def main():
    """Main function"""
    print("🚀 STUDENT DELETION TEST")
    print("=" * 60)
    
    # Create backup first
    backup_data()
    
    # Run deletion test
    test_deletion_workflow()
    
    print("\n" + "=" * 60)
    print("🎯 DELETION TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
