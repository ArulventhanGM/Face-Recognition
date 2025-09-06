"""
Advanced Face Detection Module using MTCNN and Custom CNN
Implements state-of-the-art face detection technologies
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN
import logging
from typing import List, Tuple, Optional, Dict
import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Face detection result with confidence and landmarks"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[np.ndarray] = None
    
class CustomCNN(nn.Module):
    """Custom CNN for face detection with improved architecture"""
    
    def __init__(self, num_classes=2):  # face/no-face
        super(CustomCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Bounding box regression
        self.bbox_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # x, y, width, height
        )
    
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        
        # Classification
        classification = self.classifier(flattened)
        
        # Bounding box regression
        bbox = self.bbox_regressor(flattened)
        
        return classification, bbox

class AdvancedFaceDetector:
    """Advanced face detection system using MTCNN and Custom CNN"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.mtcnn = None
        self.custom_cnn = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize detection models"""
        try:
            # Initialize MTCNN
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
                factor=0.709,
                post_process=False,
                device=self.device
            )
            logger.info("MTCNN initialized successfully")
            
            # Initialize Custom CNN
            self.custom_cnn = CustomCNN()
            
            # Try to load pre-trained weights if available
            model_path = os.path.join('models', 'custom_face_detector.pth')
            if os.path.exists(model_path):
                self.custom_cnn.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Custom CNN model loaded successfully")
            else:
                logger.warning("Custom CNN model not found, using random weights")
            
            self.custom_cnn.eval()
            
        except Exception as e:
            logger.error(f"Error initializing detection models: {e}")
    
    def detect_faces_mtcnn(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces using MTCNN"""
        try:
            if self.mtcnn is None:
                return []
            
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Detect faces
            boxes, probs, landmarks = self.mtcnn.detect(image_rgb, landmarks=True)
            
            results = []
            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    if prob > 0.9:  # High confidence threshold
                        x, y, x2, y2 = box.astype(int)
                        w, h = x2 - x, y2 - y
                        
                        results.append(DetectionResult(
                            bbox=(x, y, w, h),
                            confidence=float(prob),
                            landmarks=landmark
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in MTCNN detection: {e}")
            return []
    
    def detect_faces_custom_cnn(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces using Custom CNN with sliding window approach"""
        try:
            if self.custom_cnn is None:
                return []
            
            results = []
            h, w = image.shape[:2]
            
            # Multi-scale detection
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            window_size = 64
            
            for scale in scales:
                # Resize image
                new_h, new_w = int(h * scale), int(w * scale)
                resized = cv2.resize(image, (new_w, new_h))
                
                # Sliding window
                stride = window_size // 2
                for y in range(0, new_h - window_size + 1, stride):
                    for x in range(0, new_w - window_size + 1, stride):
                        # Extract window
                        window = resized[y:y+window_size, x:x+window_size]
                        
                        # Preprocess
                        window_tensor = self._preprocess_window(window)
                        
                        # Predict
                        with torch.no_grad():
                            classification, bbox_pred = self.custom_cnn(window_tensor)
                            
                            # Apply softmax to get probabilities
                            prob = F.softmax(classification, dim=1)[0, 1].item()  # Face probability
                            
                            if prob > 0.8:  # Confidence threshold
                                # Scale back coordinates
                                bbox_x, bbox_y, bbox_w, bbox_h = bbox_pred[0].cpu().numpy()
                                
                                # Convert relative to absolute coordinates
                                abs_x = int((x + bbox_x) / scale)
                                abs_y = int((y + bbox_y) / scale)
                                abs_w = int(bbox_w / scale)
                                abs_h = int(bbox_h / scale)
                                
                                results.append(DetectionResult(
                                    bbox=(abs_x, abs_y, abs_w, abs_h),
                                    confidence=prob
                                ))
            
            # Non-maximum suppression
            results = self._non_maximum_suppression(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Custom CNN detection: {e}")
            return []
    
    def _preprocess_window(self, window: np.ndarray) -> torch.Tensor:
        """Preprocess image window for CNN"""
        # Resize to expected input size
        window = cv2.resize(window, (64, 64))
        
        # Normalize
        window = window.astype(np.float32) / 255.0
        
        # Convert to tensor
        if len(window.shape) == 3:
            window = window.transpose(2, 0, 1)  # HWC to CHW
        else:
            window = np.expand_dims(window, 0)  # Add channel dimension
        
        window_tensor = torch.from_numpy(window).unsqueeze(0)  # Add batch dimension
        
        return window_tensor
    
    def _non_maximum_suppression(self, detections: List[DetectionResult], 
                                overlap_threshold: float = 0.3) -> List[DetectionResult]:
        """Apply non-maximum suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [det for det in detections 
                         if self._calculate_iou(best.bbox, det.bbox) < overlap_threshold]
        
        return keep
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_faces(self, image: np.ndarray, method: str = 'mtcnn') -> List[DetectionResult]:
        """Detect faces using specified method"""
        if method.lower() == 'mtcnn':
            return self.detect_faces_mtcnn(image)
        elif method.lower() == 'custom_cnn':
            return self.detect_faces_custom_cnn(image)
        elif method.lower() == 'ensemble':
            # Use both methods and combine results
            mtcnn_results = self.detect_faces_mtcnn(image)
            cnn_results = self.detect_faces_custom_cnn(image)
            
            # Combine and apply NMS
            all_results = mtcnn_results + cnn_results
            return self._non_maximum_suppression(all_results)
        else:
            logger.warning(f"Unknown detection method: {method}. Using MTCNN.")
            return self.detect_faces_mtcnn(image)
    
    def visualize_detections(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            confidence = detection.confidence
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"{confidence:.2f}"
            cv2.putText(result_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks if available
            if detection.landmarks is not None:
                for landmark in detection.landmarks:
                    cv2.circle(result_image, tuple(landmark.astype(int)), 2, (255, 0, 0), -1)
        
        return result_image

# Factory function for easy initialization
def create_face_detector(device='cpu') -> AdvancedFaceDetector:
    """Create and return an initialized face detector"""
    return AdvancedFaceDetector(device=device)
