#!/usr/bin/env python3
"""
Fix photo paths in students.csv by removing invalid photo references
"""
import pandas as pd
import os

def fix_photo_paths():
    """Remove photo path references for missing photos"""
    csv_path = 'data/students.csv'
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} student records")
    
    # Count records with photo paths
    has_photo_path = df['photo_path'].notna().sum()
    print(f"Records with photo paths: {has_photo_path}")
    
    # Check which photo paths exist
    valid_paths = []
    invalid_paths = []
    
    for idx, row in df.iterrows():
        photo_path = row.get('photo_path')
        if pd.notna(photo_path) and photo_path:
            # Convert relative path to absolute
            if not os.path.isabs(photo_path):
                full_path = os.path.join(os.getcwd(), photo_path)
            else:
                full_path = photo_path
                
            if os.path.exists(full_path):
                valid_paths.append(photo_path)
            else:
                invalid_paths.append(photo_path)
                # Clear the invalid photo path
                df.at[idx, 'photo_path'] = ''
    
    print(f"Valid photo paths: {len(valid_paths)}")
    print(f"Invalid photo paths: {len(invalid_paths)}")
    
    if invalid_paths:
        print("\nInvalid paths found:")
        for path in invalid_paths:
            print(f"  - {path}")
        
        # Save the updated CSV
        df.to_csv(csv_path, index=False)
        print(f"\nUpdated {csv_path} - cleared {len(invalid_paths)} invalid photo paths")
    else:
        print("No invalid photo paths found")

if __name__ == "__main__":
    fix_photo_paths()
