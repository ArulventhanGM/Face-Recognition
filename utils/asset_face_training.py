#!/usr/bin/env python3
"""
Asset Face Training System
Processes images from assets folder for face recognition training
Supports emotion-based training from Training/Training folder structure
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import logging
import random
import glob
import time
from typing import Dict, List, Tuple, Any, Optional
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetFaceTrainer:
    """Comprehensive asset-based face training system"""
    
    def __init__(self):
        # Use absolute paths to avoid path resolution issues
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assets_dir = os.path.join(project_root, "assets")
        
        # Main training directories
        self.training_dir = os.path.join(self.assets_dir, "Training", "Training")
        self.testing_dir = os.path.join(self.assets_dir, "Testing")
        
        # Alternative paths for different dataset structures
        self.real_images_dir = os.path.join(
            self.assets_dir, 
            "archive", 
            "Human Faces Dataset", 
            "Real Images"
        )
        
        # Models directory
        self.models_dir = os.path.join(project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Training data storage
        self.training_data = []
        self.training_labels = []
        self.label_mapping = {}
        self.trained_model = None
        
        logger.info(f"AssetFaceTrainer initialized:")
        logger.info(f"  Project root: {project_root}")
        logger.info(f"  Assets dir: {self.assets_dir}")
        logger.info(f"  Training dir: {self.training_dir}")
        logger.info(f"  Assets dir exists: {os.path.exists(self.assets_dir)}")
        logger.info(f"  Training dir exists: {os.path.exists(self.training_dir)}")
        
        # Initialize face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("✅ Face detection initialized")
        except Exception as e:
            logger.error(f"Error initializing face detection: {e}")
            self.face_cascade = None
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotion categories"""
        emotions = []
        if os.path.exists(self.training_dir):
            emotions = [d for d in os.listdir(self.training_dir) 
                       if os.path.isdir(os.path.join(self.training_dir, d))]
        return sorted(emotions)
    
    def get_emotion_image_count(self, emotion: str) -> int:
        """Get count of images for a specific emotion"""
        emotion_path = os.path.join(self.training_dir, emotion)
        if not os.path.exists(emotion_path):
            return 0
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        count = 0
        for ext in image_extensions:
            count += len(glob.glob(os.path.join(emotion_path, f"*{ext}")))
            count += len(glob.glob(os.path.join(emotion_path, f"*{ext.upper()}")))
        
        return count
    
    def get_assets_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of available assets"""
        summary = {
            'assets_available': {
                'emotions': [],
                'emotion_counts': {},
                'total_images': 0
            },
            'current_encodings': 0,
            'model_loaded': False,
            'metadata': {
                'last_training': None,
                'model_accuracy': None
            }
        }
        
        try:
            # Get emotion data
            emotions = self.get_available_emotions()
            summary['assets_available']['emotions'] = emotions
            
            total_images = 0
            for emotion in emotions:
                count = self.get_emotion_image_count(emotion)
                summary['assets_available']['emotion_counts'][emotion] = count
                total_images += count
            
            summary['assets_available']['total_images'] = total_images
            
            # Check for existing model
            model_path = os.path.join(self.models_dir, "asset_emotion_model.pkl")
            if os.path.exists(model_path):
                summary['model_loaded'] = True
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    summary['current_encodings'] = len(model_data.get('training_labels', []))
                    summary['metadata'] = model_data.get('metadata', {})
                except Exception as e:
                    logger.warning(f"Error reading model metadata: {e}")
            
        except Exception as e:
            logger.error(f"Error getting assets summary: {e}")
        
        return summary
    
    def extract_face_from_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """Extract face from image with fallback options"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Try face detection
            if self.face_cascade is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Use the largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    face_image = rgb_image[y:y+h, x:x+w]
                else:
                    # Fallback to center crop
                    h, w = rgb_image.shape[:2]
                    size = min(h, w)
                    start_y = (h - size) // 2
                    start_x = (w - size) // 2
                    face_image = rgb_image[start_y:start_y+size, start_x:start_x+size]
            else:
                # No face detection available, use full image
                face_image = rgb_image
            
            # Resize to target size
            face_image = cv2.resize(face_image, target_size)
            
            # Normalize to [0, 1]
            face_image = face_image.astype(np.float32) / 255.0
            
            return face_image
            
        except Exception as e:
            logger.warning(f"Error extracting face from {image_path}: {e}")
            return None
    
    def load_emotion_data(self, emotion: str, max_images: int = 1000) -> List[np.ndarray]:
        """Load images for a specific emotion"""
        emotion_path = os.path.join(self.training_dir, emotion)
        if not os.path.exists(emotion_path):
            logger.warning(f"Emotion directory not found: {emotion_path}")
            return []
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(glob.glob(os.path.join(emotion_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(emotion_path, f"*{ext.upper()}")))
        
        # Limit and shuffle
        if len(image_files) > max_images:
            image_files = random.sample(image_files, max_images)
        
        logger.info(f"Loading {len(image_files)} images for emotion: {emotion}")
        
        # Process images
        emotion_data = []
        failed_count = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {emotion}"):
            face_image = self.extract_face_from_image(img_path)
            if face_image is not None:
                emotion_data.append(face_image)
            else:
                failed_count += 1
        
        logger.info(f"Successfully processed {len(emotion_data)} images for {emotion} ({failed_count} failed)")
        return emotion_data
    
    def prepare_training_data(self, emotions_to_include: List[str] = None, 
                             max_images_per_emotion: int = 1000) -> Dict[str, Any]:
        """Prepare training data from emotion images"""
        logger.info("🚀 Preparing training data from assets...")
        
        # Get available emotions
        available_emotions = self.get_available_emotions()
        if not available_emotions:
            return {
                'success': False,
                'error': 'No emotion directories found in training folder',
                'training_samples': 0
            }
        
        # Use specified emotions or all available
        if emotions_to_include:
            emotions_to_use = [e for e in emotions_to_include if e in available_emotions]
        else:
            emotions_to_use = available_emotions
        
        if not emotions_to_use:
            return {
                'success': False,
                'error': 'No valid emotions selected for training',
                'training_samples': 0
            }
        
        logger.info(f"Training with emotions: {emotions_to_use}")
        
        # Load data for each emotion
        all_images = []
        all_labels = []
        emotion_counts = {}
        
        for i, emotion in enumerate(emotions_to_use):
            emotion_images = self.load_emotion_data(emotion, max_images_per_emotion)
            
            if emotion_images:
                all_images.extend(emotion_images)
                all_labels.extend([i] * len(emotion_images))
                emotion_counts[emotion] = len(emotion_images)
                
                # Create label mapping
                self.label_mapping[i] = emotion
        
        if not all_images:
            return {
                'success': False,
                'error': 'No images could be loaded from assets',
                'training_samples': 0
            }
        
        # Convert to numpy arrays
        self.training_data = np.array(all_images)
        self.training_labels = np.array(all_labels)
        
        logger.info(f"✅ Training data prepared: {len(all_images)} samples across {len(emotions_to_use)} emotions")
        
        return {
            'success': True,
            'training_samples': len(all_images),
            'emotions': emotions_to_use,
            'emotion_counts': emotion_counts,
            'label_mapping': self.label_mapping
        }
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract features from images for training"""
        logger.info("Extracting features from images...")
        
        features = []
        for i, image in enumerate(tqdm(images, desc="Extracting features")):
            try:
                # Convert back to uint8 if needed
                if image.max() <= 1.0:
                    img_uint8 = (image * 255).astype(np.uint8)
                else:
                    img_uint8 = image.astype(np.uint8)
                
                # Simple feature extraction (can be enhanced with advanced methods)
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
                
                # Resize to consistent size for feature extraction
                gray = cv2.resize(gray, (64, 64))
                
                # Calculate features
                feature_vector = []
                
                # Histogram features
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                feature_vector.extend(hist.flatten())
                
                # Statistical features
                feature_vector.append(np.mean(gray))
                feature_vector.append(np.std(gray))
                feature_vector.append(np.median(gray))
                
                # Texture features (simple)
                edges = cv2.Canny(gray, 50, 150)
                feature_vector.append(np.sum(edges > 0))
                
                # LBP-like features (simplified)
                for i in range(0, gray.shape[0]-2, 8):
                    for j in range(0, gray.shape[1]-2, 8):
                        center = gray[i+1, j+1]
                        pattern = 0
                        if gray[i, j] > center: pattern += 1
                        if gray[i, j+1] > center: pattern += 2
                        if gray[i, j+2] > center: pattern += 4
                        if gray[i+1, j+2] > center: pattern += 8
                        feature_vector.append(pattern)
                
                # Pad or truncate to fixed size
                target_size = 256
                if len(feature_vector) > target_size:
                    feature_vector = feature_vector[:target_size]
                elif len(feature_vector) < target_size:
                    feature_vector.extend([0] * (target_size - len(feature_vector)))
                
                features.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Error extracting features for image {i}: {e}")
                # Add zero vector as fallback
                features.append([0] * 256)
        
        return np.array(features)
    
    def train_emotion_classifier(self, test_size: float = 0.2) -> Dict[str, Any]:
        """Train emotion classification model"""
        logger.info("🧠 Training emotion classification model...")
        
        if len(self.training_data) == 0:
            return {
                'success': False,
                'error': 'No training data available. Please prepare data first.',
                'accuracy': 0.0
            }
        
        try:
            # Extract features
            features = self.extract_features(self.training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, self.training_labels, 
                test_size=test_size, 
                random_state=42, 
                stratify=self.training_labels
            )
            
            # Train simple classifier (can be enhanced)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.linear_model import LogisticRegression
            
            # Try multiple classifiers and use the best one
            classifiers = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            best_accuracy = 0
            best_model = None
            best_name = None
            
            logger.info("Testing different classifiers...")
            for name, clf in classifiers.items():
                try:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    logger.info(f"{name} accuracy: {accuracy:.3f}")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = clf
                        best_name = name
                        
                except Exception as e:
                    logger.warning(f"Error training {name}: {e}")
            
            if best_model is None:
                return {
                    'success': False,
                    'error': 'All classifiers failed to train',
                    'accuracy': 0.0
                }
            
            # Store the best model
            self.trained_model = {
                'classifier': best_model,
                'label_mapping': self.label_mapping,
                'model_type': best_name,
                'accuracy': best_accuracy,
                'feature_size': features.shape[1]
            }
            
            logger.info(f"✅ Best model: {best_name} with accuracy: {best_accuracy:.3f}")
            
            # Save model
            self.save_model()
            
            return {
                'success': True,
                'accuracy': best_accuracy,
                'model_type': best_name,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'emotions': list(self.label_mapping.values())
            }
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            return {
                'success': False,
                'error': str(e),
                'accuracy': 0.0
            }
    
    def save_model(self):
        """Save the trained model"""
        if self.trained_model is None:
            logger.warning("No trained model to save")
            return
        
        try:
            model_path = os.path.join(self.models_dir, "asset_emotion_model.pkl")
            
            model_data = {
                'trained_model': self.trained_model,
                'training_labels': self.training_labels.tolist() if hasattr(self.training_labels, 'tolist') else [],
                'label_mapping': self.label_mapping,
                'metadata': {
                    'last_training': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'model_accuracy': self.trained_model.get('accuracy', 0.0),
                    'total_samples': len(self.training_data) if hasattr(self, 'training_data') else 0
                }
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✅ Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self) -> bool:
        """Load a previously trained model"""
        try:
            model_path = os.path.join(self.models_dir, "asset_emotion_model.pkl")
            
            if not os.path.exists(model_path):
                logger.warning("No saved model found")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.trained_model = model_data.get('trained_model')
            self.label_mapping = model_data.get('label_mapping', {})
            
            logger.info("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_emotion(self, image_path: str) -> Dict[str, Any]:
        """Predict emotion for a given image"""
        try:
            if self.trained_model is None:
                if not self.load_model():
                    return {
                        'success': False,
                        'error': 'No trained model available'
                    }
            
            # Extract face and features
            face_image = self.extract_face_from_image(image_path)
            if face_image is None:
                return {
                    'success': False,
                    'error': 'Could not extract face from image'
                }
            
            # Extract features
            features = self.extract_features(np.array([face_image]))
            
            # Predict
            classifier = self.trained_model['classifier']
            prediction = classifier.predict(features)[0]
            probabilities = classifier.predict_proba(features)[0]
            
            # Get emotion name
            emotion = self.label_mapping.get(prediction, 'Unknown')
            confidence = float(np.max(probabilities))
            
            # Get all probabilities
            all_probs = {}
            for label_idx, prob in enumerate(probabilities):
                emotion_name = self.label_mapping.get(label_idx, f'Unknown_{label_idx}')
                all_probs[emotion_name] = float(prob)
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'all_probabilities': all_probs
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_from_assets(self, max_images_per_emotion: int = 1000, 
                         emotions_to_include: List[str] = None) -> Dict[str, Any]:
        """Main training function using asset images"""
        logger.info("🚀 Starting comprehensive asset-based training...")
        
        try:
            # Step 1: Prepare training data
            data_prep_result = self.prepare_training_data(
                emotions_to_include=emotions_to_include,
                max_images_per_emotion=max_images_per_emotion
            )
            
            if not data_prep_result['success']:
                return data_prep_result
            
            # Step 2: Train classifier
            training_result = self.train_emotion_classifier()
            
            if not training_result['success']:
                return training_result
            
            # Step 3: Combine results
            final_result = {
                'success': True,
                'message': 'Asset-based training completed successfully',
                'data_preparation': data_prep_result,
                'training_results': training_result,
                'summary': {
                    'total_samples': data_prep_result['training_samples'],
                    'emotions_trained': data_prep_result['emotions'],
                    'model_accuracy': training_result['accuracy'],
                    'model_type': training_result['model_type']
                }
            }
            
            logger.info("✅ Asset-based training completed successfully!")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in asset training: {e}")
            return {
                'success': False,
                'error': str(e),
                'processed_images': 0,
                'faces_extracted': 0
            }
    
    def check_assets_available(self) -> bool:
        """Check if training assets are available"""
        if not os.path.exists(self.training_dir):
            logger.warning(f"Training directory not found: {self.training_dir}")
            return False
        
        emotions = self.get_available_emotions()
        if not emotions:
            logger.warning("No emotion directories found")
            return False
        
        return True
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'assets_available': self.check_assets_available(),
            'emotions_count': len(self.get_available_emotions()),
            'model_trained': self.trained_model is not None,
            'total_available_images': sum(self.get_emotion_image_count(e) for e in self.get_available_emotions())
        }


def get_asset_trainer():
    """Get asset trainer instance"""
    return AssetFaceTrainer()

# For backward compatibility
def get_asset_training_system():
    """Get asset training system (backward compatibility)"""
    return get_asset_trainer()

if __name__ == "__main__":
    trainer = get_asset_trainer()
    
    # Test the trainer
    print("Available emotions:", trainer.get_available_emotions())
    print("Assets summary:", trainer.get_assets_summary())
    
    # Optionally run training
    # result = trainer.train_from_assets(max_images_per_emotion=100)
    # print(f"Training result: {result}")
