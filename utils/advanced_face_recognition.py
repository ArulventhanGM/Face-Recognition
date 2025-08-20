"""
Advanced Face Recognition System with SVM, Cross-Validation, and Data Augmentation
This module provides enhanced face recognition accuracy using machine learning techniques.
"""

import cv2
import numpy as np
import pickle
import os
import logging
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
from typing import List, Tuple, Dict, Any, Optional
import json
from datetime import datetime
import albumentations as A
from PIL import Image, ImageEnhance
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFaceRecognitionSystem:
    """Advanced Face Recognition System with ML optimization"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Model components
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None
        
        # Training data
        self.embeddings = []
        self.labels = []
        self.face_embeddings_dict = {}
        
        # Model files
        self.model_file = os.path.join(data_folder, 'advanced_face_model.pkl')
        self.scaler_file = os.path.join(data_folder, 'face_scaler.pkl')
        self.label_encoder_file = os.path.join(data_folder, 'label_encoder.pkl')
        self.pca_file = os.path.join(data_folder, 'face_pca.pkl')
        self.embeddings_file = os.path.join(data_folder, 'face_embeddings_advanced.pkl')
        self.metadata_file = os.path.join(data_folder, 'model_metadata.json')
        
        # Configuration
        self.config = {
            'face_size': (224, 224),
            'min_face_size': (50, 50),
            'confidence_threshold': 0.6,
            'use_pca': True,
            'pca_components': 0.95,  # Keep 95% variance
            'augmentation_factor': 3,  # Number of augmented versions per image
            'cv_folds': 5
        }
        
        # Load existing models if available
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            if os.path.exists(self.model_file):
                self.classifier = joblib.load(self.model_file)
                logger.info("Loaded trained classifier")
                
            if os.path.exists(self.scaler_file):
                self.scaler = joblib.load(self.scaler_file)
                logger.info("Loaded feature scaler")
                
            if os.path.exists(self.label_encoder_file):
                self.label_encoder = joblib.load(self.label_encoder_file)
                logger.info("Loaded label encoder")
                
            if os.path.exists(self.pca_file):
                self.pca = joblib.load(self.pca_file)
                logger.info("Loaded PCA transformer")
                
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.face_embeddings_dict = pickle.load(f)
                logger.info(f"Loaded {len(self.face_embeddings_dict)} face embeddings")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            os.makedirs(self.data_folder, exist_ok=True)
            
            # Save classifier
            if self.classifier:
                joblib.dump(self.classifier, self.model_file)
                
            # Save scaler
            joblib.dump(self.scaler, self.scaler_file)
            
            # Save label encoder
            joblib.dump(self.label_encoder, self.label_encoder_file)
            
            # Save PCA
            if self.pca:
                joblib.dump(self.pca, self.pca_file)
                
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.face_embeddings_dict, f)
                
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'num_identities': len(set(self.labels)) if self.labels else 0,
                'num_samples': len(self.embeddings),
                'config': self.config,
                'model_type': type(self.classifier).__name__ if self.classifier else None
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def detect_face(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image with enhanced detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # Multi-scale detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.config['min_face_size'],
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter faces with eye detection for better quality
        valid_faces = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
            
            # Accept faces with at least one eye detected or large enough faces
            if len(eyes) > 0 or (w * h) > 10000:
                valid_faces.append((x, y, w, h))
        
        return valid_faces
    
    def preprocess_face(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract and preprocess face with alignment"""
        x, y, w, h = face_coords
        
        # Extract face with padding
        padding = int(min(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
            
        # Convert to grayscale if needed
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        face = cv2.resize(face, self.config['face_size'])
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face = clahe.apply(face)
        
        # Normalize pixel values
        face = face.astype(np.float32) / 255.0
        
        return face
    
    def extract_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced features from face"""
        if face is None:
            return None
            
        # Flatten image
        features = face.flatten()
        
        # Add gradient features
        grad_x = cv2.Sobel(face, cv2.CV_32F, 1, 0, ksize=3).flatten()
        grad_y = cv2.Sobel(face, cv2.CV_32F, 0, 1, ksize=3).flatten()
        
        # Add LBP (Local Binary Pattern) features
        lbp_features = self._compute_lbp(face).flatten()
        
        # Combine all features
        combined_features = np.concatenate([features, grad_x, grad_y, lbp_features])
        
        return combined_features
    
    def _compute_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern features"""
        h, w = image.shape
        lbp_image = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                binary_pattern = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j - radius * np.sin(angle)))
                    
                    x = max(0, min(x, h-1))
                    y = max(0, min(y, w-1))
                    
                    if image[x, y] >= center:
                        binary_pattern |= (1 << k)
                
                lbp_image[i, j] = binary_pattern
        
        return lbp_image
    
    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply data augmentation to increase training samples"""
        augmented_images = [image]  # Include original
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            ], p=0.8),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(0.0, 0.01), p=0.3),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            ], p=0.4)
        ])
        
        # Generate augmented versions
        for _ in range(self.config['augmentation_factor'] - 1):
            try:
                # Convert to PIL Image for albumentations
                if len(image.shape) == 2:
                    pil_image = Image.fromarray((image * 255).astype(np.uint8), 'L')
                    pil_image = pil_image.convert('RGB')
                    img_array = np.array(pil_image)
                else:
                    img_array = (image * 255).astype(np.uint8)
                
                # Apply augmentation
                augmented = transform(image=img_array)['image']
                
                # Convert back to grayscale and normalize
                if len(augmented.shape) == 3:
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
                
                augmented = augmented.astype(np.float32) / 255.0
                augmented_images.append(augmented)
                
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                
        return augmented_images
    
    def train_face(self, student_id: str, image_paths: List[str]) -> Dict[str, Any]:
        """Train face recognition for a specific student with multiple images"""
        try:
            student_embeddings = []
            student_labels = []
            
            logger.info(f"Training face recognition for {student_id} with {len(image_paths)} images")
            
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Detect faces
                faces = self.detect_face(image)
                
                if not faces:
                    logger.warning(f"No faces detected in {image_path}")
                    continue
                
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                
                # Preprocess face
                processed_face = self.preprocess_face(image, largest_face)
                if processed_face is None:
                    continue
                
                # Apply data augmentation
                augmented_faces = self.augment_image(processed_face)
                
                for aug_face in augmented_faces:
                    # Extract features
                    features = self.extract_features(aug_face)
                    if features is not None:
                        student_embeddings.append(features)
                        student_labels.append(student_id)
            
            if not student_embeddings:
                return {
                    'success': False,
                    'message': 'No valid faces could be processed from the provided images'
                }
            
            # Store embeddings for this student
            self.face_embeddings_dict[student_id] = student_embeddings
            
            # Update global training data
            self.embeddings.extend(student_embeddings)
            self.labels.extend(student_labels)
            
            logger.info(f"Generated {len(student_embeddings)} training samples for {student_id}")
            
            return {
                'success': True,
                'message': f'Successfully trained {len(student_embeddings)} samples for {student_id}',
                'samples_count': len(student_embeddings),
                'images_processed': len([p for p in image_paths if os.path.exists(p)])
            }
            
        except Exception as e:
            logger.error(f"Error training face for {student_id}: {e}")
            return {
                'success': False,
                'message': f'Training failed: {str(e)}'
            }
    
    def optimize_classifier(self) -> Dict[str, Any]:
        """Optimize classifier using grid search and cross-validation"""
        if len(self.embeddings) < 2:
            return {'success': False, 'message': 'Not enough training data'}
        
        logger.info("Starting classifier optimization...")
        
        # Prepare data
        X = np.array(self.embeddings)
        y = np.array(self.labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if configured
        if self.config['use_pca']:
            self.pca = PCA(n_components=self.config['pca_components'])
            X_scaled = self.pca.fit_transform(X_scaled)
            logger.info(f"PCA reduced features from {X.shape[1]} to {X_scaled.shape[1]}")
        
        # Define classifiers and their hyperparameters
        classifiers = {
            'SVM': {
                'classifier': SVC(probability=True, random_state=42),
                'params': {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'classifier__kernel': ['rbf', 'poly', 'linear']
                }
            },
            'KNN': {
                'classifier': KNeighborsClassifier(),
                'params': {
                    'classifier__n_neighbors': [3, 5, 7, 9, 11],
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'RandomForest': {
                'classifier': RandomForestClassifier(random_state=42),
                'params': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        best_score = 0
        best_classifier = None
        best_params = {}
        results = {}
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42)
        
        for name, config in classifiers.items():
            logger.info(f"Optimizing {name} classifier...")
            
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('classifier', config['classifier'])
                ])
                
                # Grid search
                grid_search = GridSearchCV(
                    pipeline,
                    config['params'],
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit grid search
                grid_search.fit(X_scaled, y_encoded)
                
                # Store results
                results[name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'cv_results': grid_search.cv_results_
                }
                
                logger.info(f"{name} best score: {grid_search.best_score_:.4f}")
                
                # Update best classifier
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_classifier = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
            except Exception as e:
                logger.error(f"Error optimizing {name}: {e}")
                results[name] = {'error': str(e)}
        
        if best_classifier is None:
            return {
                'success': False,
                'message': 'Classifier optimization failed',
                'results': results
            }
        
        # Train final model with best parameters
        self.classifier = best_classifier
        self.classifier.fit(X_scaled, y_encoded)
        
        # Evaluate final model
        y_pred = self.classifier.predict(X_scaled)
        accuracy = accuracy_score(y_encoded, y_pred)
        
        # Save models
        self._save_models()
        
        logger.info(f"Model training completed with accuracy: {accuracy:.4f}")
        
        return {
            'success': True,
            'message': 'Classifier optimization completed successfully',
            'best_classifier': type(best_classifier.named_steps['classifier']).__name__,
            'best_score': best_score,
            'final_accuracy': accuracy,
            'best_params': best_params,
            'num_identities': len(np.unique(y_encoded)),
            'num_samples': len(X_scaled),
            'feature_dimensions': X_scaled.shape[1],
            'results': results
        }
    
    def predict_face(self, image: np.ndarray, min_confidence: float = None) -> Dict[str, Any]:
        """Predict identity of face in image with confidence scores"""
        if self.classifier is None:
            return {
                'success': False,
                'message': 'Model not trained. Please train the model first.'
            }
        
        min_confidence = min_confidence or self.config['confidence_threshold']
        
        try:
            # Detect faces
            faces = self.detect_face(image)
            
            if not faces:
                return {
                    'success': False,
                    'message': 'No faces detected in the image'
                }
            
            results = []
            
            for face_coords in faces:
                # Preprocess face
                processed_face = self.preprocess_face(image, face_coords)
                if processed_face is None:
                    continue
                
                # Extract features
                features = self.extract_features(processed_face)
                if features is None:
                    continue
                
                # Scale features
                features_scaled = self.scaler.transform([features])
                
                # Apply PCA if used during training
                if self.pca is not None:
                    features_scaled = self.pca.transform(features_scaled)
                
                # Predict
                prediction = self.classifier.predict(features_scaled)[0]
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                
                # Get confidence
                max_prob = np.max(probabilities)
                
                # Decode prediction
                predicted_identity = self.label_encoder.inverse_transform([prediction])[0]
                
                # Get all class probabilities
                class_probs = {}
                for i, class_name in enumerate(self.label_encoder.classes_):
                    class_probs[class_name] = float(probabilities[i])
                
                result = {
                    'face_coords': face_coords,
                    'predicted_identity': predicted_identity,
                    'confidence': float(max_prob),
                    'all_probabilities': class_probs,
                    'meets_threshold': max_prob >= min_confidence
                }
                
                results.append(result)
            
            return {
                'success': True,
                'faces_detected': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in face prediction: {e}")
            return {
                'success': False,
                'message': f'Prediction failed: {str(e)}'
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            info = {
                'is_trained': self.classifier is not None,
                'num_identities': len(self.face_embeddings_dict),
                'total_samples': sum(len(samples) for samples in self.face_embeddings_dict.values()),
                'model_type': type(self.classifier).__name__ if self.classifier else None,
                'feature_scaling': self.scaler is not None,
                'pca_enabled': self.pca is not None,
                'pca_components': self.pca.n_components_ if self.pca else None,
                'identities': list(self.face_embeddings_dict.keys()),
                'samples_per_identity': {k: len(v) for k, v in self.face_embeddings_dict.items()},
                'metadata': metadata
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}
    
    def retrain_model(self) -> Dict[str, Any]:
        """Retrain the model with current data"""
        if not self.face_embeddings_dict:
            return {
                'success': False,
                'message': 'No training data available'
            }
        
        # Rebuild training data
        self.embeddings = []
        self.labels = []
        
        for student_id, embeddings_list in self.face_embeddings_dict.items():
            self.embeddings.extend(embeddings_list)
            self.labels.extend([student_id] * len(embeddings_list))
        
        # Optimize classifier
        return self.optimize_classifier()
    
    def delete_identity(self, student_id: str) -> Dict[str, Any]:
        """Delete training data for a specific identity"""
        if student_id not in self.face_embeddings_dict:
            return {
                'success': False,
                'message': f'Identity {student_id} not found'
            }
        
        del self.face_embeddings_dict[student_id]
        
        # Retrain model if there's still data
        if self.face_embeddings_dict:
            result = self.retrain_model()
        else:
            # Clear model if no data left
            self.classifier = None
            self.embeddings = []
            self.labels = []
            result = {
                'success': True,
                'message': 'All training data deleted, model cleared'
            }
        
        return result
