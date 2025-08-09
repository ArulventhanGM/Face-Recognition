import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
import logging
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFaceRecognitionSystem:
    """Enhanced face recognition system with OpenCV Haar Cascades and improved algorithms"""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.initialize_cascades()
        logger.info("Enhanced face recognition system initialized")
    
    def initialize_cascades(self):
        """Initialize OpenCV Haar Cascades for face detection"""
        try:
            # Try to load Haar cascades from OpenCV
            cascade_path = cv2.data.haarcascades
            
            face_cascade_path = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
            eye_cascade_path = os.path.join(cascade_path, 'haarcascade_eye.xml')
            
            if os.path.exists(face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                logger.info("Haar cascade for face detection loaded successfully")
            
            if os.path.exists(eye_cascade_path):
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                logger.info("Haar cascade for eye detection loaded successfully")
                
        except Exception as e:
            logger.warning(f"Could not load Haar cascades: {e}")
            self.face_cascade = None
            self.eye_cascade = None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Enhanced face detection using OpenCV Haar Cascades"""
        try:
            if image is None or image.size == 0:
                return []
            
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            faces = []
            
            if self.face_cascade is not None:
                # Use Haar cascade detection
                detected_faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Convert to (left, top, right, bottom) format
                for (x, y, w, h) in detected_faces:
                    faces.append((x, y, x + w, y + h))
                
                # Validate faces using eye detection if available
                if self.eye_cascade is not None:
                    validated_faces = []
                    for (left, top, right, bottom) in faces:
                        face_roi = gray[top:bottom, left:right]
                        eyes = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
                        
                        # Only keep faces with at least one eye detected
                        if len(eyes) >= 1:
                            validated_faces.append((left, top, right, bottom))
                    
                    faces = validated_faces
            else:
                # Fallback: simple face detection using image properties
                h, w = gray.shape[:2]
                if h > 100 and w > 100:  # Minimum size check
                    # Assume center region contains a face
                    margin_x, margin_y = w // 4, h // 4
                    faces.append((margin_x, margin_y, w - margin_x, h - margin_y))
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def extract_face_features(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract enhanced face features from detected face region"""
        try:
            left, top, right, bottom = face_box
            
            # Extract face region
            if len(image.shape) == 3:
                face_region = image[top:bottom, left:right]
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = image[top:bottom, left:right]
            
            # Resize to standard size for consistency
            standard_size = (128, 128)
            resized_face = cv2.resize(gray_face, standard_size)
            
            # Histogram equalization for better lighting normalization
            equalized_face = cv2.equalizeHist(resized_face)
            
            # Extract multiple types of features
            features = []
            
            # 1. Raw pixel intensities (normalized)
            pixel_features = equalized_face.flatten().astype(np.float32) / 255.0
            features.extend(pixel_features)
            
            # 2. Local Binary Pattern (LBP) features
            lbp_features = self._extract_lbp_features(equalized_face)
            features.extend(lbp_features)
            
            # 3. Histogram features
            hist_features = cv2.calcHist([equalized_face], [0], None, [256], [0, 256]).flatten()
            hist_features = hist_features / (hist_features.sum() + 1e-7)  # Normalize
            features.extend(hist_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting face features: {e}")
            return np.array([])
    
    def _extract_lbp_features(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> List[float]:
        """Extract Local Binary Pattern features"""
        try:
            h, w = image.shape
            lbp_image = np.zeros_like(image)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = image[i, j]
                    binary_string = ""
                    
                    # Sample points around the center
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        
                        if 0 <= x < h and 0 <= y < w:
                            binary_string += "1" if image[x, y] >= center else "0"
                        else:
                            binary_string += "0"
                    
                    lbp_image[i, j] = int(binary_string, 2)
            
            # Calculate histogram of LBP values
            hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-7)  # Normalize
            
            return hist.tolist()
            
        except Exception as e:
            logger.error(f"Error extracting LBP features: {e}")
            return [0.0] * 256
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image using enhanced feature extraction"""
        try:
            faces = self.detect_faces(image)
            if not faces:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            
            # Extract enhanced features
            embedding = self.extract_face_features(image, largest_face)
            
            if len(embedding) > 0:
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def extract_multiple_face_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract embeddings for all faces in image"""
        try:
            faces = self.detect_faces(image)
            embeddings = []
            
            for face_box in faces:
                embedding = self.extract_face_features(image, face_box)
                if len(embedding) > 0:
                    embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting multiple face embeddings: {e}")
            return []
    
    def compare_faces(self, known_embedding: np.ndarray, unknown_embedding: np.ndarray, 
                     threshold: float = 0.7) -> Tuple[bool, float]:
        """Enhanced face comparison using multiple similarity metrics"""
        try:
            if len(known_embedding) == 0 or len(unknown_embedding) == 0:
                return False, 1.0
            
            # Ensure embeddings have the same length
            min_len = min(len(known_embedding), len(unknown_embedding))
            known_emb = known_embedding[:min_len]
            unknown_emb = unknown_embedding[:min_len]
            
            # Calculate multiple similarity metrics
            
            # 1. Cosine similarity
            cosine_sim = cosine_similarity([known_emb], [unknown_emb])[0][0]
            
            # 2. Euclidean distance (normalized)
            euclidean_dist = np.linalg.norm(known_emb - unknown_emb)
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # 3. Correlation coefficient
            correlation = np.corrcoef(known_emb, unknown_emb)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            # Combine similarities with weights
            combined_similarity = (
                0.5 * cosine_sim + 
                0.3 * euclidean_sim + 
                0.2 * abs(correlation)
            )
            
            distance = 1 - combined_similarity
            is_match = combined_similarity > threshold
            
            return is_match, distance
            
        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return False, 1.0
    
    def identify_face(self, unknown_embedding: np.ndarray, known_embeddings: Dict[str, np.ndarray], 
                     threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """Enhanced face identification with improved matching"""
        try:
            if not known_embeddings or len(unknown_embedding) == 0:
                return None, 1.0
            
            best_match = None
            best_distance = float('inf')
            best_similarity = 0.0
            
            for person_id, known_embedding in known_embeddings.items():
                is_match, distance = self.compare_faces(known_embedding, unknown_embedding, threshold)
                similarity = 1 - distance
                
                if is_match and similarity > best_similarity:
                    best_match = person_id
                    best_distance = distance
                    best_similarity = similarity
            
            return best_match, best_distance
            
        except Exception as e:
            logger.error(f"Error identifying face: {e}")
            return None, 1.0
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better recognition"""
        try:
            if image is None or image.size == 0:
                return image
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR and convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Noise reduction
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Contrast enhancement
            if len(denoised.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge channels and convert back to RGB
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Enhanced image loading with preprocessing"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Apply preprocessing
            processed_image = self.preprocess_image(image)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def process_webcam_frame(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """Process webcam frame for real-time recognition"""
        try:
            if frame is None or frame.size == 0:
                return [], []
            
            # Preprocess frame
            processed_frame = self.preprocess_image(frame)
            
            # Detect faces
            face_locations = self.detect_faces(processed_frame)
            
            # Extract embeddings for each face
            face_embeddings = []
            for face_box in face_locations:
                embedding = self.extract_face_features(processed_frame, face_box)
                if len(embedding) > 0:
                    face_embeddings.append(embedding)
            
            return face_locations, face_embeddings
            
        except Exception as e:
            logger.error(f"Error processing webcam frame: {e}")
            return [], []

# Global instance
enhanced_face_recognizer = EnhancedFaceRecognitionSystem()

def get_enhanced_face_recognizer():
    """Get the enhanced face recognizer instance"""
    return enhanced_face_recognizer
