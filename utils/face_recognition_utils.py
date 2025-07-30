import cv2
import numpy as np
import face_recognition
import insightface
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """Advanced face recognition system using ArcFace for high accuracy"""
    
    def __init__(self):
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize ArcFace model"""
        try:
            # Initialize InsightFace ArcFace model
            self.model = insightface.app.FaceAnalysis(
                providers=['CPUExecutionProvider']
            )
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("ArcFace model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ArcFace model: {e}")
            # Fallback to face_recognition library
            self.model = None
            logger.info("Using face_recognition library as fallback")
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image"""
        try:
            if self.model is not None:
                # Use ArcFace
                faces = self.model.get(image)
                if faces:
                    # Return the embedding of the largest face
                    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    return largest_face.embedding
            else:
                # Fallback to face_recognition
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    return face_encodings[0]
            
            return None
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image and return bounding boxes"""
        try:
            if self.model is not None:
                # Use ArcFace
                faces = self.model.get(image)
                return [(int(face.bbox[0]), int(face.bbox[1]), 
                        int(face.bbox[2]), int(face.bbox[3])) for face in faces]
            else:
                # Fallback to face_recognition
                face_locations = face_recognition.face_locations(image)
                # Convert from (top, right, bottom, left) to (left, top, right, bottom)
                return [(left, top, right, bottom) for (top, right, bottom, left) in face_locations]
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_multiple_face_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract embeddings for all faces in image"""
        try:
            embeddings = []
            
            if self.model is not None:
                # Use ArcFace
                faces = self.model.get(image)
                for face in faces:
                    embeddings.append(face.embedding)
            else:
                # Fallback to face_recognition
                face_encodings = face_recognition.face_encodings(image)
                embeddings.extend(face_encodings)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error extracting multiple face embeddings: {e}")
            return []
    
    def compare_faces(self, known_embedding: np.ndarray, unknown_embedding: np.ndarray, 
                     threshold: float = 0.6) -> Tuple[bool, float]:
        """Compare two face embeddings"""
        try:
            if self.model is not None:
                # Use cosine similarity for ArcFace embeddings
                similarity = cosine_similarity([known_embedding], [unknown_embedding])[0][0]
                distance = 1 - similarity
                is_match = distance < threshold
                return is_match, distance
            else:
                # Use face_recognition distance
                distance = face_recognition.face_distance([known_embedding], unknown_embedding)[0]
                is_match = distance < threshold
                return is_match, distance
        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return False, 1.0
    
    def identify_face(self, unknown_embedding: np.ndarray, known_embeddings: Dict[str, np.ndarray], 
                     threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """Identify a face against a database of known faces"""
        try:
            best_match = None
            best_distance = float('inf')
            
            for person_id, known_embedding in known_embeddings.items():
                is_match, distance = self.compare_faces(known_embedding, unknown_embedding, threshold)
                
                if is_match and distance < best_distance:
                    best_match = person_id
                    best_distance = distance
            
            return best_match, best_distance
        except Exception as e:
            logger.error(f"Error identifying face: {e}")
            return None, 1.0
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for face recognition"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image from file"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            return self.preprocess_image(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def process_webcam_frame(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """Process webcam frame for real-time recognition"""
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = self.preprocess_image(small_frame)
            
            # Detect faces and extract embeddings
            face_locations = self.detect_faces(rgb_frame)
            face_embeddings = self.extract_multiple_face_embeddings(rgb_frame)
            
            # Scale back face locations
            face_locations = [(left * 4, top * 4, right * 4, bottom * 4) 
                            for (left, top, right, bottom) in face_locations]
            
            return face_locations, face_embeddings
        except Exception as e:
            logger.error(f"Error processing webcam frame: {e}")
            return [], []

# Global instance
face_recognizer = FaceRecognitionSystem()

def get_face_recognizer():
    """Get the global face recognizer instance"""
    return face_recognizer
