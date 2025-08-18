"""
Geo-location utilities for extracting location data from images
Extracts GPS coordinates and location names from geo-tagged photos
"""

import os
import logging
from typing import Dict, Optional, Tuple, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

logger = logging.getLogger(__name__)

class GeoLocationExtractor:
    """Extract location data from geo-tagged images"""
    
    def __init__(self):
        # Initialize geocoder with a proper user agent
        self.geocoder = Nominatim(user_agent="face_recognition_attendance_system")
        self.fallback_geocoder = None
        
    def extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract all metadata from image including GPS data"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return {}
            
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                
                if not exif_data:
                    logger.info(f"No EXIF data found in image: {image_path}")
                    return {}
                
                metadata = {}
                gps_info = {}
                
                # Extract all EXIF data
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    if tag == "GPSInfo":
                        # Extract GPS information
                        for gps_tag_id, gps_value in value.items():
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            gps_info[gps_tag] = gps_value
                    else:
                        metadata[tag] = value
                
                # Add GPS info to metadata
                if gps_info:
                    metadata['GPSInfo'] = gps_info
                
                logger.info(f"Extracted metadata from {image_path}: GPS data {'found' if gps_info else 'not found'}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {image_path}: {e}")
            return {}
    
    def extract_gps_coordinates(self, metadata: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates (latitude, longitude) from metadata"""
        try:
            gps_info = metadata.get('GPSInfo', {})
            
            if not gps_info:
                return None
            
            # Check if required GPS tags are present
            required_tags = ['GPSLatitude', 'GPSLatitudeRef', 'GPSLongitude', 'GPSLongitudeRef']
            if not all(tag in gps_info for tag in required_tags):
                logger.warning("Incomplete GPS information in image")
                return None
            
            # Extract latitude
            lat_dms = gps_info['GPSLatitude']
            lat_ref = gps_info['GPSLatitudeRef']
            latitude = self._convert_dms_to_decimal(lat_dms, lat_ref)
            
            # Extract longitude
            lon_dms = gps_info['GPSLongitude']
            lon_ref = gps_info['GPSLongitudeRef']
            longitude = self._convert_dms_to_decimal(lon_dms, lon_ref)
            
            logger.info(f"Extracted GPS coordinates: {latitude}, {longitude}")
            return (latitude, longitude)
            
        except Exception as e:
            logger.error(f"Error extracting GPS coordinates: {e}")
            return None
    
    def _convert_dms_to_decimal(self, dms: Tuple, ref: str) -> float:
        """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees"""
        try:
            degrees, minutes, seconds = dms
            
            # Convert to float if needed
            degrees = float(degrees)
            minutes = float(minutes)
            seconds = float(seconds)
            
            # Calculate decimal degrees
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            # Apply reference (N/S for latitude, E/W for longitude)
            if ref.upper() in ['S', 'W']:
                decimal = -decimal
            
            return decimal
            
        except Exception as e:
            logger.error(f"Error converting DMS to decimal: {e}")
            return 0.0
    
    def get_location_name(self, latitude: float, longitude: float, timeout: int = 10) -> Dict[str, str]:
        """Get location name from coordinates using reverse geocoding"""
        try:
            location_info = {
                'address': 'Unknown Location',
                'city': '',
                'state': '',
                'country': '',
                'postal_code': '',
                'formatted_address': 'Unknown Location'
            }
            
            # Try primary geocoder
            try:
                location = self.geocoder.reverse(f"{latitude}, {longitude}", timeout=timeout)
                
                if location and location.address:
                    location_info['formatted_address'] = location.address
                    
                    # Extract detailed address components
                    if hasattr(location, 'raw') and 'address' in location.raw:
                        address_components = location.raw['address']
                        
                        # Map common address components
                        location_info['city'] = address_components.get('city', 
                                                address_components.get('town', 
                                                address_components.get('village', '')))
                        
                        location_info['state'] = address_components.get('state', 
                                               address_components.get('region', ''))
                        
                        location_info['country'] = address_components.get('country', '')
                        
                        location_info['postal_code'] = address_components.get('postcode', '')
                        
                        # Create a more user-friendly address
                        address_parts = []
                        if location_info['city']:
                            address_parts.append(location_info['city'])
                        if location_info['state']:
                            address_parts.append(location_info['state'])
                        if location_info['country']:
                            address_parts.append(location_info['country'])
                        
                        if address_parts:
                            location_info['address'] = ', '.join(address_parts)
                        else:
                            location_info['address'] = location_info['formatted_address']
                    
                    logger.info(f"Successfully geocoded location: {location_info['address']}")
                    
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.warning(f"Primary geocoder failed: {e}")
                # Try fallback methods if needed
                location_info = self._try_fallback_geocoding(latitude, longitude)
            
            return location_info
            
        except Exception as e:
            logger.error(f"Error getting location name: {e}")
            return {
                'address': 'Location Unavailable',
                'city': '',
                'state': '',
                'country': '',
                'postal_code': '',
                'formatted_address': f"Coordinates: {latitude:.6f}, {longitude:.6f}"
            }
    
    def _try_fallback_geocoding(self, latitude: float, longitude: float) -> Dict[str, str]:
        """Try alternative geocoding methods"""
        try:
            # Option 1: Try a simple HTTP request to a free geocoding service
            # Note: In production, you might want to use a proper API key
            url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={latitude}&longitude={longitude}&localityLanguage=en"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                city = data.get('city', data.get('locality', ''))
                state = data.get('principalSubdivision', '')
                country = data.get('countryName', '')
                
                address_parts = [part for part in [city, state, country] if part]
                address = ', '.join(address_parts) if address_parts else 'Unknown Location'
                
                return {
                    'address': address,
                    'city': city,
                    'state': state,
                    'country': country,
                    'postal_code': data.get('postcode', ''),
                    'formatted_address': address
                }
            
        except Exception as e:
            logger.warning(f"Fallback geocoding failed: {e}")
        
        # Final fallback - just return coordinates
        return {
            'address': f"Coordinates: {latitude:.6f}, {longitude:.6f}",
            'city': '',
            'state': '',
            'country': '',
            'postal_code': '',
            'formatted_address': f"Lat: {latitude:.6f}, Lon: {longitude:.6f}"
        }
    
    def extract_location_from_image(self, image_path: str) -> Dict[str, Any]:
        """Complete pipeline to extract location data from image"""
        try:
            result = {
                'has_location': False,
                'latitude': None,
                'longitude': None,
                'address': 'No location data',
                'city': '',
                'state': '',
                'country': '',
                'postal_code': '',
                'formatted_address': 'No location data',
                'extraction_timestamp': time.time(),
                'metadata_available': False
            }
            
            # Step 1: Extract metadata
            logger.info(f"Extracting metadata from image: {image_path}")
            metadata = self.extract_image_metadata(image_path)
            
            if not metadata:
                logger.info(f"No metadata found in image: {image_path}")
                return result
            
            result['metadata_available'] = True
            
            # Step 2: Extract GPS coordinates
            coordinates = self.extract_gps_coordinates(metadata)
            
            if not coordinates:
                logger.info(f"No GPS coordinates found in image: {image_path}")
                return result
            
            latitude, longitude = coordinates
            result['has_location'] = True
            result['latitude'] = latitude
            result['longitude'] = longitude
            
            # Step 3: Get location name
            logger.info(f"Getting location name for coordinates: {latitude}, {longitude}")
            location_info = self.get_location_name(latitude, longitude)
            
            # Update result with location information
            result.update(location_info)
            
            logger.info(f"Location extraction completed for {image_path}: {result['address']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in location extraction pipeline for {image_path}: {e}")
            return {
                'has_location': False,
                'latitude': None,
                'longitude': None,
                'address': 'Location extraction failed',
                'city': '',
                'state': '',
                'country': '',
                'postal_code': '',
                'formatted_address': 'Location extraction failed',
                'extraction_timestamp': time.time(),
                'metadata_available': False,
                'error': str(e)
            }
    
    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate GPS coordinates"""
        try:
            return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)
        except:
            return False
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in kilometers"""
        try:
            from math import radians, cos, sin, asin, sqrt
            
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            # Radius of earth in kilometers
            r = 6371
            
            return c * r
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0

# Singleton instance
_geo_extractor = None

def get_geo_location_extractor() -> GeoLocationExtractor:
    """Get singleton instance of GeoLocationExtractor"""
    global _geo_extractor
    if _geo_extractor is None:
        _geo_extractor = GeoLocationExtractor()
    return _geo_extractor
