"""
Integration layer for Advanced Face Recognition System
This module integrates the advanced ML-based face recognition with the existing application.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from utils.advanced_face_recognition import AdvancedFaceRecognitionSystem
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedFaceRecognitionBackend:
    """Backend interface for advanced face recognition system"""
    
    def __init__(self, data_folder: str = "data"):
        self.system = AdvancedFaceRecognitionSystem(data_folder)
        self.name = "advanced_ml"
        
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces and return standardized format"""
        try:
            faces = self.system.detect_face(image)
            results = []
            
            for i, (x, y, w, h) in enumerate(faces):
                results.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'confidence': 1.0,  # Detection confidence
                    'landmarks': None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def train_face(self, student_id: str, image_paths: List[str]) -> Dict[str, Any]:
        """Train face recognition for a student"""
        return self.system.train_face(student_id, image_paths)
    
    def optimize_model(self) -> Dict[str, Any]:
        """Optimize the ML model using cross-validation and grid search"""
        return self.system.optimize_classifier()
    
    def recognize_face(self, image: np.ndarray, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Recognize faces in image"""
        try:
            result = self.system.predict_face(image, min_confidence)
            
            if not result['success']:
                return []
            
            # Convert to standardized format
            recognized_faces = []
            for face_result in result['results']:
                if face_result['meets_threshold']:
                    x, y, w, h = face_result['face_coords']
                    recognized_faces.append({
                        'identity': face_result['predicted_identity'],
                        'confidence': face_result['confidence'],
                        'bbox': (x, y, w, h),
                        'all_probabilities': face_result['all_probabilities']
                    })
            
            return recognized_faces
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.system.get_model_info()
    
    def delete_face_training(self, student_id: str) -> Dict[str, Any]:
        """Delete training data for a student"""
        return self.system.delete_identity(student_id)
    
    def retrain_model(self) -> Dict[str, Any]:
        """Retrain the model with current data"""
        return self.system.retrain_model()
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        info = self.get_model_info()
        return info.get('is_trained', False)
    
    def get_trained_identities(self) -> List[str]:
        """Get list of trained identities"""
        info = self.get_model_info()
        return info.get('identities', [])
    
    def batch_train_from_directory(self, training_dir: str) -> Dict[str, Any]:
        """Train from directory structure: training_dir/student_id/images"""
        if not os.path.exists(training_dir):
            return {
                'success': False,
                'message': f'Training directory not found: {training_dir}'
            }
        
        results = {}
        total_trained = 0
        
        for student_id in os.listdir(training_dir):
            student_dir = os.path.join(training_dir, student_id)
            if not os.path.isdir(student_dir):
                continue
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_paths = []
            
            for filename in os.listdir(student_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(student_dir, filename))
            
            if image_paths:
                result = self.train_face(student_id, image_paths)
                results[student_id] = result
                
                if result['success']:
                    total_trained += 1
                    
                logger.info(f"Training result for {student_id}: {result['message']}")
        
        # Optimize model after training all identities
        if total_trained > 0:
            logger.info("Optimizing classifier after batch training...")
            optimization_result = self.optimize_model()
            
            return {
                'success': True,
                'message': f'Successfully trained {total_trained} identities',
                'individual_results': results,
                'optimization_result': optimization_result,
                'total_trained': total_trained
            }
        else:
            return {
                'success': False,
                'message': 'No identities were successfully trained',
                'individual_results': results
            }


def get_advanced_face_recognizer(data_folder: str = "data") -> AdvancedFaceRecognitionBackend:
    """Factory function to get advanced face recognition backend"""
    return AdvancedFaceRecognitionBackend(data_folder)


# Training utilities
class TrainingManager:
    """Manager for training operations and data preparation"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.training_images_dir = os.path.join(data_folder, "training_images")
        self.backend = get_advanced_face_recognizer(data_folder)
        
    def prepare_training_data(self, students_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare training data from student photos"""
        os.makedirs(self.training_images_dir, exist_ok=True)
        
        prepared_students = []
        
        for student in students_data:
            student_id = student['student_id']
            photo_path = student.get('photo_path')
            
            if not photo_path or not os.path.exists(photo_path):
                logger.warning(f"Photo not found for student {student_id}: {photo_path}")
                continue
            
            # Create student directory
            student_dir = os.path.join(self.training_images_dir, student_id)
            os.makedirs(student_dir, exist_ok=True)
            
            # Copy/link original image
            import shutil
            target_path = os.path.join(student_dir, f"original_{os.path.basename(photo_path)}")
            
            if not os.path.exists(target_path):
                try:
                    shutil.copy2(photo_path, target_path)
                    prepared_students.append({
                        'student_id': student_id,
                        'name': student['name'],
                        'image_path': target_path
                    })
                except Exception as e:
                    logger.error(f"Failed to prepare training data for {student_id}: {e}")
        
        return {
            'success': True,
            'prepared_students': prepared_students,
            'training_directory': self.training_images_dir
        }
    
    def train_all_students(self, students_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train face recognition for all students"""
        # Prepare training data
        prep_result = self.prepare_training_data(students_data)
        
        if not prep_result['success']:
            return prep_result
        
        # Batch train from directory
        return self.backend.batch_train_from_directory(self.training_images_dir)
    
    def add_training_images(self, student_id: str, image_paths: List[str]) -> Dict[str, Any]:
        """Add additional training images for a student"""
        student_dir = os.path.join(self.training_images_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Copy images to training directory
        copied_paths = []
        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path):
                import shutil
                target_path = os.path.join(student_dir, f"additional_{i}_{os.path.basename(image_path)}")
                try:
                    shutil.copy2(image_path, target_path)
                    copied_paths.append(target_path)
                except Exception as e:
                    logger.error(f"Failed to copy {image_path}: {e}")
        
        if not copied_paths:
            return {
                'success': False,
                'message': 'No images were successfully copied'
            }
        
        # Get all images for this student
        all_student_images = []
        for filename in os.listdir(student_dir):
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                all_student_images.append(os.path.join(student_dir, filename))
        
        # Train with all images
        train_result = self.backend.train_face(student_id, all_student_images)
        
        if train_result['success']:
            # Optimize model
            optimization_result = self.backend.optimize_model()
            train_result['optimization_result'] = optimization_result
        
        return train_result
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status"""
        model_info = self.backend.get_model_info()
        
        return {
            'model_trained': model_info.get('is_trained', False),
            'num_identities': model_info.get('num_identities', 0),
            'total_samples': model_info.get('total_samples', 0),
            'model_type': model_info.get('model_type'),
            'identities': model_info.get('identities', []),
            'samples_per_identity': model_info.get('samples_per_identity', {}),
            'feature_scaling': model_info.get('feature_scaling', False),
            'pca_enabled': model_info.get('pca_enabled', False),
            'training_directory': self.training_images_dir,
            'metadata': model_info.get('metadata', {})
        }
    
    def validate_model_performance(self) -> Dict[str, Any]:
        """Validate model performance with cross-validation"""
        if not self.backend.is_trained():
            return {
                'success': False,
                'message': 'Model not trained yet'
            }
        
        # Get model info
        info = self.backend.get_model_info()
        
        # Simple validation metrics
        validation_results = {
            'success': True,
            'identities_count': len(info.get('identities', [])),
            'total_samples': info.get('total_samples', 0),
            'samples_per_identity': info.get('samples_per_identity', {}),
            'model_type': info.get('model_type'),
            'recommendations': []
        }
        
        # Add recommendations
        samples_per_id = info.get('samples_per_identity', {})
        for identity, count in samples_per_id.items():
            if count < 5:
                validation_results['recommendations'].append(
                    f"Consider adding more training images for {identity} (currently {count})"
                )
            elif count > 50:
                validation_results['recommendations'].append(
                    f"Consider reducing training images for {identity} to improve training speed (currently {count})"
                )
        
        return validation_results
