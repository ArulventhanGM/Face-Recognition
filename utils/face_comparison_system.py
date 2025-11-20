"""
Face Comparison and Matching System
Provides functionality to compare real-time cropped faces with preloaded dataset images
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Optional, Any
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

class FaceComparisonSystem:
    """System for comparing real-time faces with preloaded dataset"""
    
    def __init__(self, data_manager):
        """Initialize with data manager for accessing student database"""
        self.data_manager = data_manager
        self.comparison_methods = ['histogram', 'structural_similarity', 'template_matching']
        logger.info("FaceComparisonSystem initialized")
    
    def load_student_face_images(self) -> Dict[str, Dict[str, Any]]:
        """Load all student face images for comparison"""
        try:
            students = self.data_manager.get_all_students()
            student_faces = {}
            
            for student in students:
                student_id = student['student_id']
                photo_path = student.get('photo_path')
                
                if photo_path and os.path.exists(photo_path):
                    # Load and process student image
                    student_image = cv2.imread(photo_path)
                    if student_image is not None:
                        # Detect face in student image
                        face_data = self._extract_face_from_image(student_image)
                        
                        if face_data is not None:
                            student_faces[student_id] = {
                                'student_info': student,
                                'image_path': photo_path,
                                'face_crop': face_data['face_crop'],
                                'face_embedding': face_data['embedding'],
                                'face_bbox': face_data['bbox'],
                                'quality_score': face_data['quality_score']
                            }
                            
            logger.info(f"Loaded {len(student_faces)} student face images")
            return student_faces
            
        except Exception as e:
            logger.error(f"Error loading student face images: {e}")
            return {}
    
    def _extract_face_from_image(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract face from student image"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            if len(faces) == 0:
                return None
            
            # Take the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Add padding
            padding = int(0.2 * min(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # Extract face crop
            face_crop = image[y1:y2, x1:x2]
            face_crop_resized = cv2.resize(face_crop, (150, 150))
            
            # Calculate quality score
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            quality_score = self._calculate_image_quality(gray_face)
            
            # Extract embedding
            embedding = self._extract_face_embedding(face_crop_resized)
            
            return {
                'face_crop': face_crop_resized,
                'bbox': (x, y, w, h),
                'embedding': embedding,
                'quality_score': quality_score
            }
            
        except Exception as e:
            logger.error(f"Error extracting face from image: {e}")
            return None
    
    def _calculate_image_quality(self, gray_image: np.ndarray) -> float:
        """Calculate image quality score"""
        try:
            # Laplacian variance for sharpness
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 500.0, 1.0)
            
            # Standard deviation for contrast
            contrast = min(gray_image.std() / 80.0, 1.0)
            
            # Mean for brightness
            brightness = 1.0 - abs(gray_image.mean() - 128) / 128.0
            
            return (sharpness * 0.4 + contrast * 0.3 + brightness * 0.3)
            
        except Exception as e:
            logger.warning(f"Error calculating image quality: {e}")
            return 0.5
    
    def _extract_face_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """Extract face embedding for comparison"""
        try:
            # Convert to grayscale
            if len(face_crop.shape) == 3:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_crop
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Calculate multiple features
            features = []
            
            # 1. Histogram features
            hist = cv2.calcHist([equalized], [0], None, [64], [0, 256])
            features.extend(hist.flatten() / hist.sum())
            
            # 2. LBP (Local Binary Pattern) features
            lbp = self._calculate_lbp(equalized)
            lbp_hist = cv2.calcHist([lbp], [0], None, [16], [0, 16])
            features.extend(lbp_hist.flatten() / lbp_hist.sum())
            
            # 3. Edge features
            edges = cv2.Canny(equalized, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return np.array([])
    
    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        try:
            rows, cols = image.shape
            lbp = np.zeros((rows, cols), dtype=np.uint8)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    code = 0
                    
                    for p in range(n_points):
                        angle = 2 * np.pi * p / n_points
                        x = int(round(i + radius * np.cos(angle)))
                        y = int(round(j + radius * np.sin(angle)))
                        
                        if 0 <= x < rows and 0 <= y < cols:
                            if image[x, y] >= center:
                                code |= (1 << p)
                    
                    lbp[i, j] = code
            
            return lbp
            
        except Exception as e:
            logger.warning(f"Error calculating LBP: {e}")
            return np.zeros_like(image)
    
    def find_similar_faces(self, realtime_embedding: np.ndarray, student_faces: Dict[str, Dict[str, Any]], 
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar faces from student database"""
        try:
            if realtime_embedding.size == 0:
                return []
            
            similarities = []
            
            for student_id, face_data in student_faces.items():
                student_embedding = face_data['face_embedding']
                
                if student_embedding.size == 0:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(realtime_embedding, student_embedding)
                
                similarities.append({
                    'student_id': student_id,
                    'student_info': face_data['student_info'],
                    'similarity_score': similarity,
                    'face_crop': face_data['face_crop'],
                    'image_path': face_data['image_path'],
                    'quality_score': face_data['quality_score']
                })
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top K matches
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar faces: {e}")
            return []
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between embeddings"""
        try:
            if embedding1.size == 0 or embedding2.size == 0:
                return 0.0
            
            # Ensure same length
            min_len = min(len(embedding1), len(embedding2))
            emb1 = embedding1[:min_len]
            emb2 = embedding2[:min_len]
            
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            
            # Euclidean distance (converted to similarity)
            euclidean_dist = np.linalg.norm(emb1 - emb2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            
            # Combined similarity
            combined_similarity = (cosine_sim * 0.7 + euclidean_sim * 0.3)
            
            return max(0.0, min(1.0, combined_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def create_comparison_data(self, realtime_face_path: str, similar_faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison data for UI display"""
        try:
            # Load realtime face
            realtime_image = cv2.imread(realtime_face_path)
            if realtime_image is None:
                return {'success': False, 'error': 'Failed to load realtime face'}
            
            # Convert to base64 for web display
            def image_to_base64(img: np.ndarray) -> str:
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/jpeg;base64,{img_base64}"
            
            realtime_base64 = image_to_base64(realtime_image)
            
            # Prepare similar faces data
            matches = []
            for i, face_data in enumerate(similar_faces):
                student_face_base64 = image_to_base64(face_data['face_crop'])
                
                matches.append({
                    'rank': i + 1,
                    'student_id': face_data['student_id'],
                    'student_name': face_data['student_info'].get('name', 'Unknown'),
                    'student_department': face_data['student_info'].get('department', 'Unknown'),
                    'student_year': face_data['student_info'].get('year', 'Unknown'),
                    'similarity_score': round(face_data['similarity_score'] * 100, 1),
                    'quality_score': round(face_data['quality_score'] * 100, 1),
                    'student_face_base64': student_face_base64,
                    'confidence_level': self._get_confidence_level(face_data['similarity_score'])
                })
            
            return {
                'success': True,
                'realtime_face_base64': realtime_base64,
                'matches': matches,
                'total_matches': len(matches),
                'best_match': matches[0] if matches else None
            }
            
        except Exception as e:
            logger.error(f"Error creating comparison data: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_confidence_level(self, similarity_score: float) -> str:
        """Get confidence level description"""
        if similarity_score >= 0.85:
            return 'Very High'
        elif similarity_score >= 0.75:
            return 'High'
        elif similarity_score >= 0.65:
            return 'Medium'
        elif similarity_score >= 0.5:
            return 'Low'
        else:
            return 'Very Low'
    
    def process_realtime_recognition(self, realtime_face_path: str) -> Dict[str, Any]:
        """Complete processing pipeline for realtime face recognition"""
        try:
            # Load student faces
            student_faces = self.load_student_face_images()
            
            if not student_faces:
                return {'success': False, 'error': 'No student faces loaded'}
            
            # Load and process realtime face
            realtime_image = cv2.imread(realtime_face_path)
            if realtime_image is None:
                return {'success': False, 'error': 'Failed to load realtime face'}
            
            # Extract embedding from realtime face
            realtime_embedding = self._extract_face_embedding(realtime_image)
            
            if realtime_embedding.size == 0:
                return {'success': False, 'error': 'Failed to extract features from realtime face'}
            
            # Find similar faces
            similar_faces = self.find_similar_faces(realtime_embedding, student_faces)
            
            if not similar_faces:
                return {'success': False, 'error': 'No similar faces found'}
            
            # Create comparison data
            comparison_data = self.create_comparison_data(realtime_face_path, similar_faces)
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error processing realtime recognition: {e}")
            return {'success': False, 'error': str(e)}

# Global instance
_face_comparison_system = None

def get_face_comparison_system(data_manager) -> FaceComparisonSystem:
    """Get singleton instance of face comparison system"""
    global _face_comparison_system
    if _face_comparison_system is None:
        _face_comparison_system = FaceComparisonSystem(data_manager)
    return _face_comparison_system