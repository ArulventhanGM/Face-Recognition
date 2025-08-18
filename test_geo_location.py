"""
Test script for geo-location functionality
Tests GPS data extraction from sample images
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.geo_location import get_geo_location_extractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_geo_location_extractor():
    """Test geo-location extraction with sample data"""
    print("=" * 50)
    print("Testing Geo-Location Extraction System")
    print("=" * 50)
    
    # Initialize extractor
    extractor = get_geo_location_extractor()
    
    # Test 1: Test with non-existent file
    print("\nTest 1: Non-existent file")
    result = extractor.extract_location_from_image("non_existent_file.jpg")
    print(f"Has location: {result['has_location']}")
    print(f"Address: {result['address']}")
    
    # Test 2: Test coordinate validation
    print("\nTest 2: Coordinate validation")
    valid_coords = [
        (40.7128, -74.0060),  # New York City
        (51.5074, -0.1278),   # London
        (35.6762, 139.6503),  # Tokyo
        (-33.8688, 151.2093), # Sydney
    ]
    
    invalid_coords = [
        (100, 50),      # Invalid latitude
        (45, 200),      # Invalid longitude
        (-100, 0),      # Invalid latitude
        (0, -200),      # Invalid longitude
    ]
    
    print("Valid coordinates:")
    for lat, lon in valid_coords:
        is_valid = extractor.validate_coordinates(lat, lon)
        print(f"  {lat}, {lon}: {'Valid' if is_valid else 'Invalid'}")
    
    print("Invalid coordinates:")
    for lat, lon in invalid_coords:
        is_valid = extractor.validate_coordinates(lat, lon)
        print(f"  {lat}, {lon}: {'Valid' if is_valid else 'Invalid'}")
    
    # Test 3: Test reverse geocoding with known coordinates
    print("\nTest 3: Reverse geocoding test")
    test_locations = [
        (40.7589, -73.9851, "Times Square, NYC"),
        (51.5007, -0.1246, "London Bridge"),
        (48.8584, 2.2945, "Eiffel Tower"),
    ]
    
    for lat, lon, expected_location in test_locations:
        print(f"\nTesting: {expected_location} ({lat}, {lon})")
        location_info = extractor.get_location_name(lat, lon)
        print(f"  Address: {location_info['address']}")
        print(f"  City: {location_info['city']}")
        print(f"  Country: {location_info['country']}")
        print(f"  Formatted: {location_info['formatted_address']}")
    
    # Test 4: Test distance calculation
    print("\nTest 4: Distance calculations")
    distances = [
        ((40.7589, -73.9851), (40.7505, -73.9934), "Times Square to Empire State Building"),
        ((51.5007, -0.1246), (51.5074, -0.1278), "London Bridge to Big Ben"),
        ((0, 0), (0, 1), "1 degree longitude at equator"),
    ]
    
    for (lat1, lon1), (lat2, lon2), description in distances:
        distance = extractor.calculate_distance(lat1, lon1, lat2, lon2)
        print(f"  {description}: {distance:.2f} km")
    
    # Test 5: Test with sample image (if available)
    print("\nTest 5: Sample image processing")
    
    # Look for sample images in common directories
    sample_paths = [
        "test_images",
        "samples",
        "data/test",
        "uploads"
    ]
    
    sample_image = None
    for path in sample_paths:
        test_dir = project_root / path
        if test_dir.exists():
            for ext in ['jpg', 'jpeg', 'png', 'tiff']:
                sample_files = list(test_dir.glob(f"*.{ext}"))
                if sample_files:
                    sample_image = sample_files[0]
                    break
            if sample_image:
                break
    
    if sample_image and sample_image.exists():
        print(f"Testing with sample image: {sample_image}")
        result = extractor.extract_location_from_image(str(sample_image))
        print(f"  Has location: {result['has_location']}")
        print(f"  Metadata available: {result['metadata_available']}")
        if result['has_location']:
            print(f"  Coordinates: {result['latitude']}, {result['longitude']}")
            print(f"  Address: {result['address']}")
        else:
            print("  No GPS data found in image")
    else:
        print("No sample images found for testing")
        print("To test with real images, place geo-tagged photos in one of these directories:")
        for path in sample_paths:
            print(f"  - {project_root / path}")
    
    print("\n" + "=" * 50)
    print("Geo-Location Testing Complete!")
    print("=" * 50)

def create_test_image_with_metadata():
    """Create a sample image with fake GPS metadata for testing"""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        import piexif
        
        print("\nCreating test image with GPS metadata...")
        
        # Create a simple test image
        test_image = Image.new('RGB', (200, 200), color='blue')
        
        # Create GPS data (Times Square coordinates)
        lat = 40.7589
        lon = -73.9851
        
        # Convert to DMS format
        def decimal_to_dms(decimal):
            degrees = int(abs(decimal))
            minutes_float = (abs(decimal) - degrees) * 60
            minutes = int(minutes_float)
            seconds = (minutes_float - minutes) * 60
            return [(degrees, 1), (minutes, 1), (int(seconds * 10000), 10000)]
        
        lat_dms = decimal_to_dms(lat)
        lon_dms = decimal_to_dms(lon)
        
        lat_ref = 'N' if lat >= 0 else 'S'
        lon_ref = 'E' if lon >= 0 else 'W'
        
        gps_dict = {
            piexif.GPSIFD.GPSLatitude: lat_dms,
            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
            piexif.GPSIFD.GPSLongitude: lon_dms,
            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        }
        
        exif_dict = {"GPS": gps_dict}
        exif_bytes = piexif.dump(exif_dict)
        
        # Save test image
        test_dir = project_root / "test_images"
        test_dir.mkdir(exist_ok=True)
        test_image_path = test_dir / "test_with_gps.jpg"
        
        test_image.save(str(test_image_path), exif=exif_bytes)
        print(f"Created test image with GPS data: {test_image_path}")
        
        # Test the created image
        extractor = get_geo_location_extractor()
        result = extractor.extract_location_from_image(str(test_image_path))
        
        print(f"Test image results:")
        print(f"  Has location: {result['has_location']}")
        if result['has_location']:
            print(f"  Coordinates: {result['latitude']}, {result['longitude']}")
            print(f"  Address: {result['address']}")
        
    except ImportError as e:
        print(f"Cannot create test image: {e}")
        print("Install piexif to create test images: pip install piexif")
    except Exception as e:
        print(f"Error creating test image: {e}")

if __name__ == "__main__":
    try:
        test_geo_location_extractor()
        
        # Optionally create test image
        create_test = input("\nWould you like to create a test image with GPS data? (y/n): ").lower().strip()
        if create_test == 'y':
            create_test_image_with_metadata()
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
