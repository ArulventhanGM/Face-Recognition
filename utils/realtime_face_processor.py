"""
Enhanced Real-time Face Recognition System with Face Cropping and Comparison
Provides advanced face detection, cropping, and comparison capabilities for attendance marking
"""

import cv2
import numpy as np
import base64
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

class RealTimeFaceProcessor:
    """Enhanced real-time face processing with cropping and comparison capabilities"""
    
    def __init__(self):
        """Initialize the real-time face processor"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Face detection parameters
        self.min_face_size = (30, 30)
        self.max_face_size = (300, 300)
        self.scale_factor = 1.1
        self.min_neighbors = 5
        
        # Temporary storage for cropped faces
        self.temp_dir = tempfile.mkdtemp(prefix="face_crops_")
        self.crop_counter = 0
        
        logger.info("RealTimeFaceProcessor initialized successfully")
    
    def detect_faces_enhanced(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Enhanced face detection with confidence scoring and quality assessment
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detected faces with metadata
        """
        try:
            if frame is None or frame.size == 0:
                return []
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better detection
            gray = cv2.equalizeHist(gray)
            
            # Detect faces with multiple scale factors for better accuracy
            faces_1 = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=6,
                minSize=self.min_face_size,
                maxSize=self.max_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            faces_2 = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=4,
                minSize=self.min_face_size,
                maxSize=self.max_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Combine and filter faces
            all_faces = np.vstack((faces_1, faces_2)) if len(faces_1) > 0 and len(faces_2) > 0 else faces_1 if len(faces_1) > 0 else faces_2
            
            if len(all_faces) == 0:
                return []
            
            # Remove duplicate detections using NMS
            faces = self._apply_nms(all_faces, 0.3)
            
            # Process each detected face
            face_data = []
            for i, (x, y, w, h) in enumerate(faces):
                face_info = self._analyze_face_quality(frame, gray, (x, y, w, h), i)
                if face_info['quality_score'] > 0.3:  # Filter low-quality faces
                    face_data.append(face_info)
            
            # Sort by quality score (best first)
            face_data.sort(key=lambda x: x['quality_score'], reverse=True)
            
            return face_data
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def _apply_nms(self, faces: np.ndarray, threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if len(faces) == 0:
            return []
        
        # Convert to (x1, y1, x2, y2) format
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        boxes = np.array(boxes, dtype=np.float32)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by bottom-right y-coordinate
        indices = np.argsort(boxes[:, 3])
        
        keep = []
        while len(indices) > 0:
            # Take the last index
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)
            
            # Find overlapping boxes
            xx1 = np.maximum(boxes[i, 0], boxes[indices[:last], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[indices[:last], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[indices[:last], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[indices[:last], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[indices[:last]] - intersection
            overlap = intersection / union
            
            # Remove overlapping boxes
            indices = np.delete(indices, np.concatenate(([last], np.where(overlap > threshold)[0])))
        
        # Return as (x, y, w, h)
        result = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i].astype(int)
            result.append((x1, y1, x2 - x1, y2 - y1))
        
        return result
    
    def _analyze_face_quality(self, frame: np.ndarray, gray: np.ndarray, face_box: Tuple[int, int, int, int], face_id: int) -> Dict[str, Any]:
        """Analyze face quality and extract metadata"""
        x, y, w, h = face_box
        
        # Extract face region
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        # Calculate quality metrics
        quality_score = self._calculate_face_quality(face_gray)
        
        # Detect eyes for additional verification
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
        has_eyes = len(eyes) >= 2
        
        # Calculate face area and aspect ratio
        face_area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Check if face is well-positioned (not too close to edges)
        h_frame, w_frame = frame.shape[:2]
        edge_distance = min(x, y, w_frame - (x + w), h_frame - (y + h))
        is_well_positioned = edge_distance > 20
        
        face_info = {
            'id': face_id,
            'bbox': (x, y, w, h),
            'center': (x + w//2, y + h//2),
            'quality_score': quality_score,
            'has_eyes': has_eyes,
            'eye_count': len(eyes),
            'face_area': face_area,
            'aspect_ratio': aspect_ratio,
            'is_well_positioned': is_well_positioned,
            'edge_distance': edge_distance,
            'timestamp': time.time(),
            'face_region_gray': face_gray,
            'face_region_color': face_color
        }
        
        return face_info
    
    def _calculate_face_quality(self, face_gray: np.ndarray) -> float:
        """Calculate face quality score based on multiple factors"""
        if face_gray.size == 0:
            return 0.0
        
        try:
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize
            
            # 2. Contrast (standard deviation)
            contrast_score = min(face_gray.std() / 80.0, 1.0)  # Normalize
            
            # 3. Brightness (mean intensity)
            brightness = face_gray.mean()
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # Optimal around 128
            
            # 4. Size adequacy
            h, w = face_gray.shape
            size_score = min(min(w, h) / 50.0, 1.0)  # Prefer faces > 50px
            
            # Combined quality score
            quality_score = (
                sharpness_score * 0.3 +
                contrast_score * 0.25 +
                brightness_score * 0.25 +
                size_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error calculating face quality: {e}")
            return 0.5
    
    def crop_and_save_face(self, frame: np.ndarray, face_info: Dict[str, Any], prefix: str = "realtime") -> str:
        """
        Crop face from frame and save to temporary file
        
        Args:
            frame: Source video frame
            face_info: Face detection information
            prefix: Filename prefix
            
        Returns:
            Path to saved cropped face image
        """
        try:
            x, y, w, h = face_info['bbox']
            
            # Add padding around face
            padding = int(0.2 * min(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            # Extract face region
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Resize to standard size
            face_crop = cv2.resize(face_crop, (150, 150))
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{prefix}_face_{face_info['id']}_{timestamp}.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, face_crop)
            
            # Update counter
            self.crop_counter += 1
            
            logger.info(f"Saved cropped face: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error cropping face: {e}")
            return None
    
    def get_face_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """Extract face embedding for comparison"""
        try:
            if face_crop.size == 0:
                return np.array([])
            
            # Convert to grayscale if needed
            if len(face_crop.shape) == 3:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_crop
            
            # Resize to standard size
            resized = cv2.resize(gray, (100, 100))
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(resized)
            
            # Calculate histogram as simple embedding
            hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])
            embedding = hist.flatten() / hist.sum()  # Normalize
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return np.array([])
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two face embeddings"""
        try:
            if embedding1.size == 0 or embedding2.size == 0:
                return 0.0
            
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def cleanup_temp_files(self, max_age_minutes: int = 30):
        """Clean up old temporary files"""
        try:
            current_time = time.time()
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    file_age = (current_time - os.path.getmtime(filepath)) / 60
                    if file_age > max_age_minutes:
                        os.remove(filepath)
                        logger.debug(f"Cleaned up old temp file: {filename}")
        except Exception as e:
            logger.warning(f"Error cleaning temp files: {e}")
    
    def process_realtime_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Complete processing pipeline for real-time frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Processing results with detected faces and cropped images
        """
        try:
            # Detect faces
            detected_faces = self.detect_faces_enhanced(frame)
            
            # Process each detected face
            processed_faces = []
            for face_info in detected_faces:
                # Crop and save face
                crop_path = self.crop_and_save_face(frame, face_info)
                
                if crop_path:
                    # Extract embedding for comparison
                    face_crop = cv2.imread(crop_path)
                    embedding = self.get_face_embedding(face_crop)
                    
                    # Add crop information to face info
                    face_info['crop_path'] = crop_path
                    face_info['embedding'] = embedding
                    face_info['crop_size'] = face_crop.shape if face_crop is not None else (0, 0, 0)
                    
                processed_faces.append(face_info)
            
            # Clean up old files periodically
            if self.crop_counter % 50 == 0:
                self.cleanup_temp_files()
            
            return {
                'success': True,
                'frame_shape': frame.shape,
                'faces_detected': len(detected_faces),
                'faces_processed': len(processed_faces),
                'faces': processed_faces,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error processing realtime frame: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'faces_processed': 0,
                'faces': []
            }

# Global instance
_realtime_processor = None

def get_realtime_processor() -> RealTimeFaceProcessor:
    """Get singleton instance of real-time face processor"""
    global _realtime_processor
    if _realtime_processor is None:
        _realtime_processor = RealTimeFaceProcessor()
    return _realtime_processor