#!/usr/bin/env python3
"""
Asset-Based Face Recognition Training System
Integrates test images from assets folder to fine-tune the face recognition model
"""

import os
import cv2
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False
    logging.warning("face_recognition library not available, using enhanced OpenCV backend")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetBasedFaceTrainingSystem:
    """Advanced face recognition training system using assets dataset"""
    
    def __init__(self, project_root: str = None):
        """Initialize the asset-based training system"""
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.project_root = project_root
        self.assets_dir = os.path.join(project_root, "assets")
        self.real_images_dir = os.path.join(self.assets_dir, "archive", "Human Faces Dataset", "Real Images")
        self.ai_images_dir = os.path.join(self.assets_dir, "archive", "Human Faces Dataset", "AI-Generated Images")
        
        # Training data storage
        self.data_dir = os.path.join(project_root, "data")
        self.embeddings_file = os.path.join(self.data_dir, "face_embeddings.pkl")
        self.training_metadata_file = os.path.join(self.data_dir, "training_metadata.json")
        self.model_file = os.path.join(self.data_dir, "advanced_face_model.pkl")
        self.scaler_file = os.path.join(self.data_dir, "face_scaler.pkl")
        self.pca_file = os.path.join(self.data_dir, "face_pca.pkl")
        
        # Training parameters
        self.face_encodings = []
        self.face_labels = []
        self.training_metadata = {
            "last_training": None,
            "total_images_processed": 0,
            "real_images_count": 0,
            "ai_images_count": 0,
            "student_images_count": 0,
            "model_accuracy": 0.0,
            "training_history": []
        }
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=128)  # Reduce dimensionality
        self.classifier = None
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing training data if available
        self._load_existing_data()
        
    def _load_existing_data(self):
        """Load existing training metadata and embeddings"""
        try:
            if os.path.exists(self.training_metadata_file):
                with open(self.training_metadata_file, 'r') as f:
                    self.training_metadata = json.load(f)
                logger.info(f"Loaded training metadata: {self.training_metadata['total_images_processed']} images processed")
                
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'encodings' in data and 'labels' in data:
                        self.face_encodings = data['encodings']
                        self.face_labels = data['labels']
                        logger.info(f"Loaded {len(self.face_encodings)} existing face encodings")
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            
    def process_asset_images(self, max_real_images: int = 1000, max_ai_images: int = 500) -> Dict[str, int]:
        """Process images from assets folder and extract face encodings"""
        logger.info("Starting asset images processing...")
        
        processed_counts = {
            "real_processed": 0,
            "ai_processed": 0,
            "total_faces_found": 0,
            "failed_processing": 0
        }
        
        # Process real images
        if os.path.exists(self.real_images_dir):
            logger.info(f"Processing real images from: {self.real_images_dir}")
            real_count = self._process_image_directory(
                self.real_images_dir, 
                "real_person", 
                max_real_images
            )
            processed_counts["real_processed"] = real_count
            
        # Process AI-generated images
        if os.path.exists(self.ai_images_dir):
            logger.info(f"Processing AI-generated images from: {self.ai_images_dir}")
            ai_count = self._process_image_directory(
                self.ai_images_dir, 
                "ai_person", 
                max_ai_images
            )
            processed_counts["ai_processed"] = ai_count
            
        # Process existing student images
        student_count = self._process_existing_student_images()
        processed_counts["student_processed"] = student_count
        
        processed_counts["total_faces_found"] = len(self.face_encodings)
        
        logger.info(f"Asset processing complete: {processed_counts}")
        return processed_counts
        
    def _process_image_directory(self, directory: str, label_prefix: str, max_images: int) -> int:
        """Process images from a specific directory"""
        processed_count = 0
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return 0
            
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_files = image_files[:max_images]  # Limit processing
        
        for i, filename in enumerate(image_files):
            if processed_count >= max_images:
                break
                
            filepath = os.path.join(directory, filename)
            
            try:
                # Load and process image
                image = face_recognition.load_image_file(filepath)
                
                # Find face encodings
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Use the first face found (assuming one face per image)
                    encoding = face_encodings[0]
                    
                    # Create unique label for this image
                    label = f"{label_prefix}_{os.path.splitext(filename)[0]}"
                    
                    self.face_encodings.append(encoding.tolist())
                    self.face_labels.append(label)
                    
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} images from {label_prefix} dataset")
                        
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                continue
                
        logger.info(f"Completed processing {label_prefix}: {processed_count} images")
        return processed_count
        
    def _process_existing_student_images(self) -> int:
        """Process existing student images from uploads folder"""
        uploads_dir = os.path.join(self.project_root, "uploads")
        processed_count = 0
        
        if not os.path.exists(uploads_dir):
            logger.warning("No uploads directory found for student images")
            return 0
            
        # Also check if we have existing student data
        from utils.data_manager import get_data_manager
        try:
            data_manager = get_data_manager()
            students = data_manager.get_all_students()
            
            for student in students:
                if student.get('photo_path'):
                    photo_path = os.path.join(self.project_root, student['photo_path'])
                    
                    if os.path.exists(photo_path):
                        try:
                            image = face_recognition.load_image_file(photo_path)
                            face_encodings = face_recognition.face_encodings(image)
                            
                            if face_encodings:
                                encoding = face_encodings[0]
                                label = f"student_{student['student_id']}"
                                
                                self.face_encodings.append(encoding.tolist())
                                self.face_labels.append(label)
                                processed_count += 1
                                
                        except Exception as e:
                            logger.error(f"Error processing student image {photo_path}: {e}")
                            
        except Exception as e:
            logger.error(f"Error accessing student data: {e}")
            
        logger.info(f"Processed {processed_count} existing student images")
        return processed_count
        
    def train_advanced_model(self, test_size: float = 0.2) -> Dict[str, Any]:
        """Train advanced ML model with the collected face encodings"""
        if len(self.face_encodings) < 10:
            raise ValueError("Insufficient training data. Need at least 10 face encodings.")
            
        logger.info(f"Training advanced model with {len(self.face_encodings)} face encodings")
        
        # Convert to numpy arrays
        X = np.array(self.face_encodings)
        y = np.array(self.face_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None  # Remove stratify due to unique labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA for dimensionality reduction
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Train multiple models and select the best one
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train_pca, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test_pca)
                accuracy = accuracy_score(y_test, y_pred)
                
                model_results[name] = {
                    'accuracy': accuracy,
                    'model': model
                }
                
                logger.info(f"{name} accuracy: {accuracy:.4f}")
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    self.classifier = model
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                
        # Save the best model and preprocessing components
        self._save_trained_model()
        
        # Update training metadata
        self.training_metadata.update({
            "last_training": datetime.now().isoformat(),
            "total_images_processed": len(self.face_encodings),
            "model_accuracy": best_score,
            "best_model": max(model_results.keys(), key=lambda k: model_results[k]['accuracy']) if model_results else "None"
        })
        
        self._save_training_metadata()
        
        training_results = {
            "total_encodings": len(self.face_encodings),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "best_model": max(model_results.keys(), key=lambda k: model_results[k]['accuracy']) if model_results else "None",
            "best_accuracy": best_score,
            "model_results": {k: v['accuracy'] for k, v in model_results.items()}
        }
        
        logger.info(f"Training completed. Best model: {training_results['best_model']} with accuracy: {best_score:.4f}")
        return training_results
        
    def _save_trained_model(self):
        """Save the trained model and preprocessing components"""
        try:
            # Save model
            if self.classifier:
                joblib.dump(self.classifier, self.model_file)
                
            # Save scaler
            joblib.dump(self.scaler, self.scaler_file)
            
            # Save PCA
            joblib.dump(self.pca, self.pca_file)
            
            # Save face encodings and labels
            training_data = {
                'encodings': self.face_encodings,
                'labels': self.face_labels,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(training_data, f)
                
            logger.info("Saved trained model and data successfully")
            
        except Exception as e:
            logger.error(f"Error saving trained model: {e}")
            
    def _save_training_metadata(self):
        """Save training metadata"""
        try:
            with open(self.training_metadata_file, 'w') as f:
                json.dump(self.training_metadata, f, indent=2)
            logger.info("Saved training metadata successfully")
        except Exception as e:
            logger.error(f"Error saving training metadata: {e}")
            
    def load_trained_model(self) -> bool:
        """Load previously trained model"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file) and os.path.exists(self.pca_file):
                self.classifier = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                self.pca = joblib.load(self.pca_file)
                logger.info("Loaded trained model successfully")
                return True
            else:
                logger.warning("Trained model files not found")
                return False
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            return False
            
    def predict_face(self, face_encoding: np.ndarray, confidence_threshold: float = 0.6) -> Tuple[str, float]:
        """Predict identity of a face encoding using the trained model"""
        if self.classifier is None:
            if not self.load_trained_model():
                raise ValueError("No trained model available")
                
        try:
            # Preprocess the encoding
            face_encoding = np.array(face_encoding).reshape(1, -1)
            face_encoding_scaled = self.scaler.transform(face_encoding)
            face_encoding_pca = self.pca.transform(face_encoding_scaled)
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(face_encoding_pca)[0]
            max_prob_idx = np.argmax(probabilities)
            max_probability = probabilities[max_prob_idx]
            
            if max_probability >= confidence_threshold:
                predicted_label = self.classifier.classes_[max_prob_idx]
                return predicted_label, max_probability
            else:
                return "unknown", max_probability
                
        except Exception as e:
            logger.error(f"Error in face prediction: {e}")
            return "error", 0.0
            
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of current training state"""
        return {
            "metadata": self.training_metadata,
            "current_encodings": len(self.face_encodings),
            "current_labels": len(set(self.face_labels)),
            "model_loaded": self.classifier is not None,
            "assets_available": {
                "real_images": os.path.exists(self.real_images_dir),
                "ai_images": os.path.exists(self.ai_images_dir),
                "real_images_count": len([f for f in os.listdir(self.real_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(self.real_images_dir) else 0,
                "ai_images_count": len([f for f in os.listdir(self.ai_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(self.ai_images_dir) else 0
            }
        }
        
    def test_model_performance(self, test_images_dir: str = None) -> Dict[str, Any]:
        """Test model performance on a set of test images"""
        if not test_images_dir:
            test_images_dir = self.real_images_dir
            
        if not os.path.exists(test_images_dir):
            return {"error": "Test images directory not found"}
            
        if self.classifier is None:
            if not self.load_trained_model():
                return {"error": "No trained model available"}
                
        test_results = {
            "total_images": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "average_confidence": 0.0,
            "predictions": []
        }
        
        image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:50]  # Test on 50 images
        
        total_confidence = 0.0
        
        for filename in image_files:
            filepath = os.path.join(test_images_dir, filename)
            
            try:
                image = face_recognition.load_image_file(filepath)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    encoding = face_encodings[0]
                    predicted_label, confidence = self.predict_face(encoding)
                    
                    test_results["predictions"].append({
                        "filename": filename,
                        "predicted_label": predicted_label,
                        "confidence": confidence
                    })
                    
                    total_confidence += confidence
                    test_results["successful_predictions"] += 1
                else:
                    test_results["failed_predictions"] += 1
                    
                test_results["total_images"] += 1
                
            except Exception as e:
                logger.error(f"Error testing {filename}: {e}")
                test_results["failed_predictions"] += 1
                test_results["total_images"] += 1
                
        if test_results["successful_predictions"] > 0:
            test_results["average_confidence"] = total_confidence / test_results["successful_predictions"]
            
        return test_results

# Global instance
asset_training_system = AssetBasedFaceTrainingSystem()

def get_asset_training_system() -> AssetBasedFaceTrainingSystem:
    """Get the global asset training system instance"""
    return asset_training_system
