#!/usr/bin/env python3
"""
Migration script to update existing student records with photo_path column
and attempt to link existing photos to students
"""

import os
import pandas as pd
import shutil
from datetime import datetime
from utils.data_manager import DataManager
from utils.security import SecureCSVHandler

def migrate_student_photos():
    """Migrate existing student data to include photo_path column"""
    
    print("🔄 Starting student photo migration...")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Read existing students data
    try:
        df = SecureCSVHandler.safe_read_csv(data_manager.students_file)
        print(f"📊 Found {len(df)} existing student records")
        
        # Check if photo_path column already exists
        if 'photo_path' in df.columns:
            print("✅ photo_path column already exists")
            # Check for empty photo paths and try to link existing photos
            empty_photo_paths = df[df['photo_path'].isna() | (df['photo_path'] == '')]
            if len(empty_photo_paths) > 0:
                print(f"🔍 Found {len(empty_photo_paths)} students without photo paths, attempting to link existing photos...")
                link_existing_photos(df, data_manager)
            else:
                print("✅ All students already have photo paths assigned")
        else:
            print("➕ Adding photo_path column to student records...")
            # Add photo_path column with empty values
            df['photo_path'] = ''
            
            # Try to link existing photos
            link_existing_photos(df, data_manager)
            
            # Save updated data
            SecureCSVHandler.safe_write_csv(df.to_dict('records'), data_manager.students_file, data_manager.student_columns)
            print("✅ Migration completed successfully")
            
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False
    
    return True

def link_existing_photos(df, data_manager):
    """Attempt to link existing photos to students"""
    
    upload_folder = "uploads"  # Default upload folder
    if not os.path.exists(upload_folder):
        print(f"⚠️ Upload folder '{upload_folder}' not found")
        return
    
    linked_count = 0
    
    # Get list of image files in upload folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    
    try:
        for filename in os.listdir(upload_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        
        print(f"🖼️ Found {len(image_files)} image files in upload folder")
        
        # Try to match photos to students
        for index, student in df.iterrows():
            student_id = student['student_id']
            current_photo_path = student.get('photo_path', '')
            
            # Skip if student already has a valid photo path
            if current_photo_path and os.path.exists(current_photo_path):
                continue
            
            # Look for photos that might belong to this student
            matched_photo = None
            
            # Method 1: Look for exact student ID match
            for photo_file in image_files:
                if student_id.lower() in photo_file.lower():
                    matched_photo = os.path.join(upload_folder, photo_file)
                    break
            
            # Method 2: Look for student name match (if no ID match found)
            if not matched_photo:
                student_name_parts = student['name'].lower().split()
                for photo_file in image_files:
                    photo_lower = photo_file.lower()
                    if any(name_part in photo_lower for name_part in student_name_parts if len(name_part) > 2):
                        matched_photo = os.path.join(upload_folder, photo_file)
                        break
            
            if matched_photo and os.path.exists(matched_photo):
                df.at[index, 'photo_path'] = matched_photo
                linked_count += 1
                print(f"🔗 Linked {student_id} ({student['name']}) to {os.path.basename(matched_photo)}")
        
        print(f"✅ Successfully linked {linked_count} photos to students")
        
    except Exception as e:
        print(f"❌ Error linking photos: {e}")

def verify_photo_links():
    """Verify that photo links are working correctly"""
    
    print("\n🔍 Verifying photo links...")
    
    data_manager = DataManager()
    students = data_manager.get_all_students()
    
    valid_photos = 0
    missing_photos = 0
    
    for student in students:
        student_id = student['student_id']
        photo_path = student.get('photo_path', '')
        
        if photo_path and str(photo_path) != 'nan' and os.path.exists(str(photo_path)):
            valid_photos += 1
            print(f"✅ {student_id}: Photo found at {photo_path}")
        else:
            missing_photos += 1
            print(f"❌ {student_id}: No photo found (path: '{photo_path}')")
    
    print(f"\n📊 Verification Results:")
    print(f"   ✅ Students with valid photos: {valid_photos}")
    print(f"   ❌ Students with missing photos: {missing_photos}")
    print(f"   📈 Photo coverage: {(valid_photos / len(students) * 100):.1f}%" if students else "No students found")

def create_backup():
    """Create backup of current student data"""
    
    data_manager = DataManager()
    backup_file = f"data/students_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        if os.path.exists(data_manager.students_file):
            shutil.copy2(data_manager.students_file, backup_file)
            print(f"💾 Backup created: {backup_file}")
            return True
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return False
    
    return False

def main():
    """Main migration function"""
    
    print("🚀 Student Photo Migration Tool")
    print("=" * 50)
    
    # Create backup first
    if not create_backup():
        print("⚠️ Could not create backup, proceeding anyway...")
    
    # Run migration
    if migrate_student_photos():
        # Verify results
        verify_photo_links()
        print("\n🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("1. Test the photo display in the web interface")
        print("2. Upload new photos for students without existing photos")
        print("3. Verify that new student registrations work correctly")
    else:
        print("\n❌ Migration failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
