"""
Mock face recognition utilities for testing without heavy dependencies
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class MockFaceRecognitionSystem:
    """Mock face recognition system for testing"""
    
    def __init__(self):
        self.model = "mock"
        logger.info("Mock face recognition system initialized")
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Mock face embedding extraction with deterministic results"""
        if image is not None and image.size > 0:
            # Create a deterministic embedding based on image properties
            # This ensures the same image produces the same embedding
            image_hash = hash(image.tobytes()) % 1000000
            np.random.seed(image_hash)  # Seed with image hash for consistency
            embedding = np.random.rand(512).astype(np.float32)
            np.random.seed()  # Reset seed
            return embedding
        return None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Mock face detection"""
        if image is not None and image.size > 0:
            # Return a mock face location
            h, w = image.shape[:2]
            return [(w//4, h//4, 3*w//4, 3*h//4)]
        return []
    
    def extract_multiple_face_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """Mock multiple face embeddings with deterministic results"""
        faces = self.detect_faces(image)
        embeddings = []
        for i, face in enumerate(faces):
            # Create deterministic embedding for each face
            face_hash = hash(image.tobytes() + str(i).encode()) % 1000000
            np.random.seed(face_hash)
            embedding = np.random.rand(512).astype(np.float32)
            np.random.seed()  # Reset seed
            embeddings.append(embedding)
        return embeddings
    
    def compare_faces(self, known_embedding: np.ndarray, unknown_embedding: np.ndarray,
                     threshold: float = 0.6) -> Tuple[bool, float]:
        """Mock face comparison with realistic similarity"""
        # Calculate actual distance between embeddings for more realistic results
        distance = np.linalg.norm(known_embedding - unknown_embedding)
        # Normalize distance to 0-1 range
        distance = min(distance / 10.0, 1.0)
        is_match = distance < threshold
        return is_match, distance
    
    def identify_face(self, unknown_embedding: np.ndarray, known_embeddings: Dict[str, np.ndarray], 
                     threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """Mock face identification"""
        if not known_embeddings:
            return None, 1.0
        
        # Simulate finding a match 70% of the time
        if np.random.random() < 0.7:
            student_id = list(known_embeddings.keys())[0]
            distance = np.random.uniform(0.2, 0.5)
            return student_id, distance
        
        return None, 1.0
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Mock image preprocessing"""
        return image
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Mock image loading"""
        try:
            # Create a mock image for testing
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        except:
            return None
    
    def process_webcam_frame(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """Mock webcam frame processing"""
        face_locations = self.detect_faces(frame)
        face_embeddings = self.extract_multiple_face_embeddings(frame)
        return face_locations, face_embeddings

# Global instance
face_recognizer = MockFaceRecognitionSystem()

def get_face_recognizer():
    """Get the mock face recognizer instance"""
    return face_recognizer
