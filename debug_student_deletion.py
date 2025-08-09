#!/usr/bin/env python3
"""
Debug script to investigate student deletion issues
"""

import os
import sys
import pandas as pd
import pickle
from utils.data_manager import get_data_manager
from utils.security import SecureCSVHandler

def analyze_student_data():
    """Analyze the current student data"""
    print("🔍 ANALYZING STUDENT DATA")
    print("=" * 50)
    
    # Read students CSV
    try:
        df = pd.read_csv('data/students.csv')
        print(f"✅ Students CSV loaded: {len(df)} records")
        print("\nStudent Records:")
        for idx, row in df.iterrows():
            print(f"  {idx+1}. ID: '{row['student_id']}' | Name: '{row['name']}' | Photo: '{row.get('photo_path', 'N/A')}'")
    except Exception as e:
        print(f"❌ Error reading students CSV: {e}")
        return False
    
    # Check face embeddings
    try:
        with open('data/face_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        print(f"\n✅ Face embeddings loaded: {len(embeddings)} records")
        print("Face Embedding Keys:")
        for key in embeddings.keys():
            print(f"  - '{key}'")
    except Exception as e:
        print(f"❌ Error reading face embeddings: {e}")
    
    # Check photo files
    print(f"\n📁 Photo Files in uploads/:")
    if os.path.exists('uploads'):
        files = os.listdir('uploads')
        for file in files:
            print(f"  - {file}")
    else:
        print("  No uploads directory found")
    
    return True

def test_student_lookup():
    """Test student lookup for problematic IDs"""
    print("\n🔍 TESTING STUDENT LOOKUP")
    print("=" * 50)
    
    problematic_ids = ['123456789', '987654321', '9876542134', '98765432134']
    
    data_manager = get_data_manager()
    
    for student_id in problematic_ids:
        print(f"\nTesting Student ID: '{student_id}'")
        student = data_manager.get_student(student_id)
        if student:
            print(f"  ✅ Found: {student['name']} ({student['email']})")
            print(f"  Photo: {student.get('photo_path', 'None')}")
        else:
            print(f"  ❌ Not found")

def test_deletion_process():
    """Test the deletion process step by step"""
    print("\n🗑️ TESTING DELETION PROCESS")
    print("=" * 50)
    
    # Test with the actual student IDs from CSV
    test_ids = ['123456789', '987654321', '98765432134']
    
    data_manager = get_data_manager()
    
    for student_id in test_ids:
        print(f"\n--- Testing deletion for Student ID: '{student_id}' ---")
        
        # Step 1: Check if student exists
        student = data_manager.get_student(student_id)
        if not student:
            print(f"  ❌ Student not found - cannot delete")
            continue
        
        print(f"  ✅ Student found: {student['name']}")
        
        # Step 2: Check dependencies
        print("  🔍 Checking dependencies:")
        
        # Check face embeddings
        if student_id in data_manager.face_embeddings:
            print(f"    - Face embedding: EXISTS")
        else:
            print(f"    - Face embedding: None")
        
        # Check photo file
        photo_path = student.get('photo_path', '')
        if photo_path and photo_path != 'nan' and os.path.exists(photo_path):
            print(f"    - Photo file: EXISTS ({photo_path})")
        else:
            print(f"    - Photo file: None or missing")
        
        # Step 3: Simulate deletion (without actually deleting)
        print("  🧪 Simulating deletion process:")
        
        try:
            # Read CSV
            df = SecureCSVHandler.safe_read_csv(data_manager.students_file)
            original_count = len(df)
            print(f"    - Original student count: {original_count}")
            
            # Filter out the student
            df_filtered = df[df['student_id'] != student_id]
            new_count = len(df_filtered)
            print(f"    - After filtering: {new_count}")
            
            if new_count == original_count - 1:
                print(f"    ✅ CSV filtering would work correctly")
            else:
                print(f"    ❌ CSV filtering issue - expected {original_count-1}, got {new_count}")
                
        except Exception as e:
            print(f"    ❌ Error in deletion simulation: {e}")

def check_csv_data_types():
    """Check CSV data types and potential issues"""
    print("\n📊 CHECKING CSV DATA TYPES")
    print("=" * 50)
    
    try:
        df = pd.read_csv('data/students.csv')
        print("Data types:")
        print(df.dtypes)
        
        print(f"\nStudent ID column analysis:")
        print(f"  - Type: {df['student_id'].dtype}")
        print(f"  - Sample values: {list(df['student_id'].values)}")
        print(f"  - Unique values: {df['student_id'].nunique()}")
        
        # Check for any NaN or problematic values
        print(f"\nData quality check:")
        print(f"  - Null values: {df.isnull().sum().sum()}")
        print(f"  - Duplicate student IDs: {df['student_id'].duplicated().sum()}")
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

def main():
    """Main function"""
    print("🚀 STUDENT DELETION DEBUG ANALYSIS")
    print("=" * 60)
    
    # Change to the correct directory
    if not os.path.exists('data/students.csv'):
        print("❌ students.csv not found. Make sure you're in the correct directory.")
        return
    
    # Run all analysis
    analyze_student_data()
    test_student_lookup()
    check_csv_data_types()
    test_deletion_process()
    
    print("\n" + "=" * 60)
    print("📋 ANALYSIS COMPLETE")
    print("=" * 60)
    print("Review the output above to identify deletion issues.")

if __name__ == "__main__":
    main()
