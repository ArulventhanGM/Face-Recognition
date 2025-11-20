"""
Integrated Advanced Face Recognition System
Combines MTCNN/Custom CNN detection, Custom CNN+ArcFace recognition, and advanced matching
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import time

# Import our advanced modules
from .advanced_face_detection import AdvancedFaceDetector, DetectionResult, create_face_detector
from .advanced_face_recognition import AdvancedFaceRecognizer, RecognitionResult, create_face_recognizer
from .advanced_face_matching import AdvancedFaceMatcher, MatchResult, create_face_matcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FaceProcessingResult:
    """Complete face processing result"""
    detection_results: List[DetectionResult]
    recognition_results: List[RecognitionResult]
    match_results: List[MatchResult]
    processing_time: float
    image_with_annotations: Optional[np.ndarray] = None

class IntegratedFaceSystem:
    """Integrated advanced face recognition system"""
    
    def __init__(self, 
                 detection_method: str = 'mtcnn',
                 embedding_size: int = 512,
                 matching_method: str = 'ensemble',
                 device: str = 'cpu'):
        """
        Initialize the integrated face recognition system
        
        Args:
            detection_method: 'mtcnn', 'custom_cnn', or 'ensemble'
            embedding_size: Size of face embeddings
            matching_method: 'cosine_similarity', 'euclidean_distance', 'svm_classifier', or 'ensemble'
            device: 'cpu' or 'cuda'
        """
        self.detection_method = detection_method
        self.embedding_size = embedding_size
        self.matching_method = matching_method
        self.device = device
        
        # Initialize components
        self.detector = None
        self.recognizer = None
        self.matcher = None
        
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing advanced face recognition system...")
            
            # Initialize face detector
            self.detector = create_face_detector(device=self.device)
            logger.info(f"Face detector initialized with method: {self.detection_method}")
            
            # Initialize face recognizer
            self.recognizer = create_face_recognizer(
                embedding_size=self.embedding_size,
                device=self.device
            )
            logger.info(f"Face recognizer initialized with embedding size: {self.embedding_size}")
            
            # Initialize face matcher
            self.matcher = create_face_matcher(embedding_dim=self.embedding_size)
            logger.info(f"Face matcher initialized with method: {self.matching_method}")
            
            logger.info("Advanced face recognition system initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise
    
    def register_face(self, image: np.ndarray, identity: str) -> bool:
        """Register a new face in the system"""
        try:
            # Detect faces in the image
            detections = self.detector.detect_faces(image, method=self.detection_method)
            
            if not detections:
                logger.warning(f"No faces detected for {identity}")
                return False
            
            # Use the largest detected face
            largest_detection = max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
            x, y, w, h = largest_detection.bbox
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            # Extract embedding
            embedding = self.recognizer.extract_embedding(face_region)
            if embedding is None:
                logger.error(f"Failed to extract embedding for {identity}")
                return False
            
            # Add to recognizer's known faces
            self.recognizer.add_known_face(face_region, identity)
            
            # Add to matcher's database
            self.matcher.add_face_embedding(embedding, identity)
            
            logger.info(f"Successfully registered face for {identity}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering face for {identity}: {e}")
            return False
    
    def process_image(self, image: np.ndarray, 
                     annotate: bool = True,
                     confidence_threshold: float = 0.7) -> FaceProcessingResult:
        """Process an image through the complete pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Face Detection
            detections = self.detector.detect_faces(image, method=self.detection_method)
            
            recognition_results = []
            match_results = []
            
            # Step 2: Process each detected face
            for detection in detections:
                x, y, w, h = detection.bbox
                
                # Extract face region with some padding
                padding = 20
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(image.shape[1], x + w + padding)
                y_end = min(image.shape[0], y + h + padding)
                
                face_region = image[y_start:y_end, x_start:x_end]
                
                # Step 3: Face Recognition (Extract Embedding)
                embedding = self.recognizer.extract_embedding(face_region)
                
                if embedding is not None:
                    # Create recognition result
                    recognition_result = RecognitionResult(
                        embedding=embedding,
                        confidence=detection.confidence,
                        bbox=detection.bbox
                    )
                    recognition_results.append(recognition_result)
                    
                    # Step 4: Face Matching
                    match_result = self.matcher.match_face(
                        embedding,
                        method=self.matching_method,
                        threshold=confidence_threshold
                    )
                    match_results.append(match_result)
                else:
                    # Create empty results for failed recognition
                    recognition_results.append(RecognitionResult(
                        embedding=np.array([]),
                        confidence=0.0,
                        bbox=detection.bbox
                    ))
                    match_results.append(MatchResult(
                        identity="Recognition Failed",
                        confidence=0.0,
                        similarity_score=0.0,
                        distance=float('inf'),
                        method_used="none"
                    ))
            
            # Step 5: Annotate image if requested
            annotated_image = None
            if annotate:
                annotated_image = self._annotate_image(image, detections, match_results)
            
            processing_time = time.time() - start_time
            
            return FaceProcessingResult(
                detection_results=detections,
                recognition_results=recognition_results,
                match_results=match_results,
                processing_time=processing_time,
                image_with_annotations=annotated_image
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            processing_time = time.time() - start_time
            return FaceProcessingResult(
                detection_results=[],
                recognition_results=[],
                match_results=[],
                processing_time=processing_time
            )
    
    def _annotate_image(self, image: np.ndarray, 
                       detections: List[DetectionResult],
                       matches: List[MatchResult]) -> np.ndarray:
        """Annotate image with detection and recognition results"""
        annotated = image.copy()
        
        for detection, match in zip(detections, matches):
            x, y, w, h = detection.bbox
            
            # Choose color based on confidence
            if match.confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif match.confidence > 0.5:
                color = (0, 165, 255)  # Orange - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence or unknown
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label = f"{match.identity}"
            confidence_label = f"{match.confidence:.2f}"
            method_label = f"({match.method_used})"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(confidence_label, font, font_scale, thickness)
            (method_w, method_h), _ = cv2.getTextSize(method_label, font, font_scale - 0.2, 1)
            
            max_width = max(text_w, conf_w, method_w)
            total_height = text_h + conf_h + method_h + 10
            
            # Draw background rectangle
            cv2.rectangle(annotated, (x, y - total_height - 5), 
                         (x + max_width + 10, y), color, -1)
            
            # Draw text
            cv2.putText(annotated, label, (x + 5, y - total_height + text_h),
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(annotated, confidence_label, (x + 5, y - total_height + text_h + conf_h + 5),
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(annotated, method_label, (x + 5, y - 5),
                       font, font_scale - 0.2, (255, 255, 255), 1)
            
            # Draw landmarks if available
            if detection.landmarks is not None:
                for landmark in detection.landmarks:
                    cv2.circle(annotated, tuple(landmark.astype(int)), 2, (255, 0, 0), -1)
        
        return annotated
    
    def train_system(self, training_data: List[Tuple[np.ndarray, str]], 
                    train_recognizer: bool = True,
                    train_matcher: bool = True,
                    **kwargs) -> Dict[str, float]:
        """Train the system components"""
        results = {}
        
        try:
            if train_recognizer:
                logger.info("Training face recognizer...")
                self.recognizer.train_model(training_data, **kwargs)
                results['recognizer_trained'] = True
            
            if train_matcher:
                logger.info("Training face matcher...")
                
                # Prepare embeddings for matcher training
                for image, identity in training_data:
                    # Process image to get embedding
                    result = self.process_image(image, annotate=False)
                    if result.recognition_results:
                        embedding = result.recognition_results[0].embedding
                        if len(embedding) > 0:
                            self.matcher.add_face_embedding(embedding, identity)
                
                # Train SVM classifier
                accuracy = self.matcher.train_svm_classifier()
                results['matcher_accuracy'] = accuracy
            
            logger.info("System training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error training system: {e}")
            return {'error': str(e)}
    
    def save_system(self, base_path: str):
        """Save all system components"""
        try:
            os.makedirs(base_path, exist_ok=True)
            
            # Save recognizer
            recognizer_path = os.path.join(base_path, 'recognizer.pth')
            self.recognizer.save_model(recognizer_path)
            
            # Save matcher
            matcher_path = os.path.join(base_path, 'matcher.pkl')
            self.matcher.save_matcher(matcher_path)
            
            logger.info(f"System saved to {base_path}")
            
        except Exception as e:
            logger.error(f"Error saving system: {e}")
    
    def load_system(self, base_path: str):
        """Load all system components"""
        try:
            # Load recognizer
            recognizer_path = os.path.join(base_path, 'recognizer.pth')
            if os.path.exists(recognizer_path):
                self.recognizer.load_model(recognizer_path)
            
            # Load matcher
            matcher_path = os.path.join(base_path, 'matcher.pkl')
            if os.path.exists(matcher_path):
                self.matcher.load_matcher(matcher_path)
            
            logger.info(f"System loaded from {base_path}")
            
        except Exception as e:
            logger.error(f"Error loading system: {e}")
    
    def get_system_info(self) -> Dict[str, Union[str, int, float, bool]]:
        """Get information about the system"""
        try:
            matcher_stats = self.matcher.get_statistics()
            
            return {
                'detection_method': self.detection_method,
                'embedding_size': self.embedding_size,
                'matching_method': self.matching_method,
                'device': self.device,
                'total_registered_faces': matcher_stats.get('total_embeddings', 0),
                'unique_identities': matcher_stats.get('unique_identities', 0),
                'svm_trained': matcher_stats.get('is_svm_trained', False),
                'system_ready': self.detector is not None and 
                               self.recognizer is not None and 
                               self.matcher is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {'error': str(e)}

# Factory function for easy initialization
def create_integrated_system(detection_method: str = 'mtcnn',
                           embedding_size: int = 512,
                           matching_method: str = 'ensemble',
                           device: str = 'cpu') -> IntegratedFaceSystem:
    """Create and return an initialized integrated face recognition system"""
    return IntegratedFaceSystem(
        detection_method=detection_method,
        embedding_size=embedding_size,
        matching_method=matching_method,
        device=device
    )
