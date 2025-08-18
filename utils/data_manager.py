import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from utils.security import SecureCSVHandler, validate_student_id, validate_email
import pickle
import cv2
import logging

# ---------------------------------------------------------------------------
# Logging setup (guarded so repeated imports don't duplicate handlers)
# ---------------------------------------------------------------------------
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Face recognizer backend selection with robust fallbacks.
# We purposefully keep the imports local & guarded so a missing heavy dependency
# (e.g. face_recognition / dlib / insightface) does not break app startup.
# ---------------------------------------------------------------------------
FACE_RECOGNITION_BACKEND = "unknown"

def _select_face_recognizer():
    global FACE_RECOGNITION_BACKEND
    # Attempt primary (advanced) stack first
    try:
        from utils.face_recognition_utils import get_face_recognizer  # type: ignore
        FACE_RECOGNITION_BACKEND = "arcface|face_recognition"
        logger.info("Face recognition backend: face_recognition_utils (%s)", FACE_RECOGNITION_BACKEND)
        return get_face_recognizer
    except Exception as e_primary:  # Catch broad to ensure resilience
        logger.warning("Primary face_recognition_utils import failed: %s", e_primary)

    # Attempt enhanced OpenCV-based system
    try:
        from utils.enhanced_face_recognition import get_enhanced_face_recognizer as get_face_recognizer  # type: ignore
        FACE_RECOGNITION_BACKEND = "enhanced_opencv"
        logger.info("Face recognition backend: enhanced_face_recognition (%s)", FACE_RECOGNITION_BACKEND)
        return get_face_recognizer
    except Exception as e_enhanced:
        logger.warning("Enhanced face recognition import failed: %s", e_enhanced)

    # Final fallback: lightweight deterministic mock for development/tests
    try:
        from utils.face_recognition_mock import get_face_recognizer  # type: ignore
        FACE_RECOGNITION_BACKEND = "mock"
        logger.info("Face recognition backend: mock (%s)", FACE_RECOGNITION_BACKEND)
        return get_face_recognizer
    except Exception as e_mock:
        # Absolute last resort: create a stub so rest of code does not crash
        logger.error("All face recognition backends failed: %s", e_mock)

        class _StubRecognizer:
            def load_image(self, *_a, **_k):
                return None
            def extract_face_embedding(self, *_a, **_k):
                return None
            def extract_multiple_face_embeddings(self, *_a, **_k):
                return []
            def detect_faces(self, *_a, **_k):
                return []
            def identify_face(self, *_a, **_k):
                return (None, 1.0)
            def process_webcam_frame(self, *_a, **_k):
                return [], []

        FACE_RECOGNITION_BACKEND = "stub"
        def _get_stub():
            return _StubRecognizer()
        return _get_stub

# Provide selected backend getter (late binding keeps module import cheap)
get_face_recognizer = _select_face_recognizer()

def get_face_recognition_backend() -> str:
    """Expose which backend is active (for diagnostics / UI)."""
    return FACE_RECOGNITION_BACKEND

class DataManager:
    """Secure data manager for student information and attendance"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.students_file = os.path.join(data_folder, "students.csv")
        self.attendance_file = os.path.join(data_folder, "attendance.csv")
        self.embeddings_file = os.path.join(data_folder, "face_embeddings.pkl")
        
        # CSV column definitions
        self.student_columns = ['student_id', 'name', 'email', 'department', 'year', 'registration_date', 'photo_path']
        self.attendance_columns = ['student_id', 'name', 'date', 'time', 'status', 'confidence', 'method', 
                                  'latitude', 'longitude', 'location', 'city', 'state', 'country']
        
        # In-memory storage for face embeddings
        self.face_embeddings = {}
        
        # Initialize data files
        self._initialize_data_files()
        self._load_face_embeddings()
    
    def _initialize_data_files(self):
        """Initialize CSV files if they don't exist"""
        os.makedirs(self.data_folder, exist_ok=True)
        
        # Initialize students file
        if not os.path.exists(self.students_file):
            SecureCSVHandler.safe_write_csv([], self.students_file, self.student_columns)
        
        # Initialize attendance file
        if not os.path.exists(self.attendance_file):
            SecureCSVHandler.safe_write_csv([], self.attendance_file, self.attendance_columns)
    
    def _load_face_embeddings(self):
        """Load face embeddings from file"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.face_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.face_embeddings)} face embeddings")
            else:
                self.face_embeddings = {}
        except Exception as e:
            logger.error(f"Error loading face embeddings: {e}")
            self.face_embeddings = {}
    
    def _save_face_embeddings(self):
        """Save face embeddings to file"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.face_embeddings, f)
            logger.info(f"Saved {len(self.face_embeddings)} face embeddings")
        except Exception as e:
            logger.error(f"Error saving face embeddings: {e}")
    
    def add_student(self, student_data: Dict[str, Any], face_image_path: Optional[str] = None) -> bool:
        """Add a new student with optional face data"""
        try:
            # Validate required fields
            if not validate_student_id(student_data.get('student_id', '')):
                raise ValueError("Invalid student ID format")
            
            if not validate_email(student_data.get('email', '')):
                raise ValueError("Invalid email format")
            
            # Check if student already exists
            if self.get_student(student_data['student_id']):
                raise ValueError("Student ID already exists")
            
            # Add registration date
            student_data['registration_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Store photo path
            student_data['photo_path'] = face_image_path if face_image_path else ''

            # Extract face embedding if image provided
            if face_image_path and os.path.exists(face_image_path):
                face_recognizer = get_face_recognizer()
                image = face_recognizer.load_image(face_image_path)
                if image is not None:
                    embedding = face_recognizer.extract_face_embedding(image)
                    if embedding is not None:
                        self.face_embeddings[student_data['student_id']] = embedding
                        self._save_face_embeddings()
                    else:
                        logger.warning(f"Could not extract face embedding for student {student_data['student_id']}")

            # Add to CSV
            SecureCSVHandler.append_to_csv(student_data, self.students_file, self.student_columns)
            
            logger.info(f"Added student: {student_data['student_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding student: {e}")
            return False
    
    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get student information by ID"""
        try:
            df = SecureCSVHandler.safe_read_csv(self.students_file)
            student_row = df[df['student_id'] == student_id]
            
            if not student_row.empty:
                return student_row.iloc[0].to_dict()
            
            return None
        except Exception as e:
            logger.error(f"Error getting student {student_id}: {e}")
            return None
    
    def get_all_students(self) -> List[Dict[str, Any]]:
        """Get all students"""
        try:
            df = SecureCSVHandler.safe_read_csv(self.students_file)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting all students: {e}")
            return []
    
    def update_student(self, student_id: str, updated_data: Dict[str, Any], new_photo_path: Optional[str] = None) -> bool:
        """Update student information"""
        try:
            df = SecureCSVHandler.safe_read_csv(self.students_file)

            # Find student
            student_index = df[df['student_id'] == student_id].index
            if student_index.empty:
                return False

            # Update data
            for key, value in updated_data.items():
                if key in self.student_columns and key != 'student_id':  # Don't allow ID changes
                    df.loc[student_index[0], key] = value

            # Update photo path if provided
            if new_photo_path is not None:
                df.loc[student_index[0], 'photo_path'] = new_photo_path

            # Save back to CSV
            SecureCSVHandler.safe_write_csv(df.to_dict('records'), self.students_file, self.student_columns)

            logger.info(f"Updated student: {student_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating student {student_id}: {e}")
            return False
    
    def delete_student(self, student_id: str) -> bool:
        """Delete student and associated data"""
        try:
            # Get student info before deletion for photo cleanup
            student = self.get_student(student_id)

            df = SecureCSVHandler.safe_read_csv(self.students_file)

            # Remove student from DataFrame
            df = df[df['student_id'] != student_id]

            # Save back to CSV
            SecureCSVHandler.safe_write_csv(df.to_dict('records'), self.students_file, self.student_columns)

            # Remove face embedding
            if student_id in self.face_embeddings:
                del self.face_embeddings[student_id]
                self._save_face_embeddings()

            # Clean up photo file if it exists
            if student and student.get('photo_path'):
                photo_path = student['photo_path']
                # Check if photo_path is valid (not NaN, not 'nan', not empty)
                if (photo_path and
                    str(photo_path).lower() != 'nan' and
                    str(photo_path).strip() != '' and
                    not pd.isna(photo_path)):
                    if os.path.exists(str(photo_path)):
                        try:
                            os.remove(str(photo_path))
                            logger.info(f"Deleted photo file: {photo_path}")
                        except Exception as photo_error:
                            logger.warning(f"Could not delete photo file {photo_path}: {photo_error}")

            logger.info(f"Deleted student: {student_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting student {student_id}: {e}")
            return False
    
    def mark_attendance(self, student_id: str, method: str = "camera", confidence: float = 0.0, 
                       location_data: Optional[Dict[str, Any]] = None) -> bool:
        """Mark attendance for a student with optional location data"""
        try:
            student = self.get_student(student_id)
            if not student:
                logger.error(f"Student {student_id} not found")
                return False
            
            current_time = datetime.now()
            attendance_data = {
                'student_id': student_id,
                'name': student['name'],
                'date': current_time.strftime('%Y-%m-%d'),
                'time': current_time.strftime('%H:%M:%S'),
                'status': 'Present',
                'confidence': f"{confidence:.2f}",
                'method': method
            }
            
            # Add location data if available
            if location_data and location_data.get('has_location', False):
                attendance_data.update({
                    'latitude': str(location_data.get('latitude', '')),
                    'longitude': str(location_data.get('longitude', '')),
                    'location': location_data.get('address', 'Unknown Location'),
                    'city': location_data.get('city', ''),
                    'state': location_data.get('state', ''),
                    'country': location_data.get('country', '')
                })
                logger.info(f"Added location data for {student_id}: {location_data.get('address', 'Unknown')}")
            else:
                # Add empty location fields
                attendance_data.update({
                    'latitude': '',
                    'longitude': '',
                    'location': 'No location data',
                    'city': '',
                    'state': '',
                    'country': ''
                })
            
            # Check if already marked today
            today_attendance = self.get_attendance_by_date(current_time.strftime('%Y-%m-%d'))
            already_marked = any(att['student_id'] == student_id for att in today_attendance)
            
            if already_marked:
                logger.warning(f"Attendance already marked for {student_id} today")
                return False
            
            # Add to CSV
            SecureCSVHandler.append_to_csv(attendance_data, self.attendance_file, self.attendance_columns)
            
            logger.info(f"Marked attendance for student: {student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error marking attendance for {student_id}: {e}")
            return False
    
    def get_attendance_by_date(self, date: str) -> List[Dict[str, Any]]:
        """Get attendance records for a specific date"""
        try:
            df = SecureCSVHandler.safe_read_csv(self.attendance_file)
            date_attendance = df[df['date'] == date]
            return date_attendance.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting attendance for date {date}: {e}")
            return []
    
    def get_all_attendance(self) -> List[Dict[str, Any]]:
        """Get all attendance records"""
        try:
            df = SecureCSVHandler.safe_read_csv(self.attendance_file)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error getting all attendance: {e}")
            return []
    
    def recognize_face_from_image(self, image_path: str) -> Dict[str, Any]:
        """Recognize faces from uploaded image"""
        try:
            face_recognizer = get_face_recognizer()
            image = face_recognizer.load_image(image_path)

            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image',
                    'faces_detected': 0,
                    'faces_recognized': [],
                    'faces_unrecognized': []
                }

            # Detect faces first
            face_locations = face_recognizer.detect_faces(image)

            if len(face_locations) == 0:
                return {
                    'success': True,
                    'error': None,
                    'faces_detected': 0,
                    'faces_recognized': [],
                    'faces_unrecognized': [],
                    'message': 'No faces detected in the image'
                }

            # Extract all face embeddings from image
            face_embeddings = face_recognizer.extract_multiple_face_embeddings(image)

            recognized_faces = []
            unrecognized_faces = []

            for i, embedding in enumerate(face_embeddings):
                # Try to identify the face with optimized threshold
                # Use higher threshold for better accuracy (0.75 instead of default 0.6)
                student_id, distance = face_recognizer.identify_face(embedding, self.face_embeddings, threshold=0.75)

                location = face_locations[i] if i < len(face_locations) else None

                if student_id:
                    student = self.get_student(student_id)
                    if student:
                        confidence = 1 - distance  # Convert distance to confidence

                        recognized_faces.append({
                            'student_id': student_id,
                            'name': student['name'],
                            'email': student['email'],
                            'department': student['department'],
                            'year': student['year'],
                            'confidence': confidence,
                            'location': location
                        })
                    else:
                        # Student ID found but student data missing
                        unrecognized_faces.append({
                            'face_index': i,
                            'location': location,
                            'reason': 'Student data not found'
                        })
                else:
                    # Face detected but not recognized
                    unrecognized_faces.append({
                        'face_index': i,
                        'location': location,
                        'reason': 'Face not in database'
                    })

            return {
                'success': True,
                'error': None,
                'faces_detected': len(face_locations),
                'faces_recognized': recognized_faces,
                'faces_unrecognized': unrecognized_faces,
                'message': f'Detected {len(face_locations)} face(s), recognized {len(recognized_faces)}'
            }

        except Exception as e:
            logger.error(f"Error recognizing faces from image: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'faces_recognized': [],
                'faces_unrecognized': []
            }
    
    def recognize_face_from_embedding(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Recognize face from embedding (for real-time recognition)"""
        try:
            face_recognizer = get_face_recognizer()
            # Use optimized threshold for real-time recognition
            student_id, distance = face_recognizer.identify_face(embedding, self.face_embeddings, threshold=0.75)

            if student_id:
                student = self.get_student(student_id)
                if student:
                    confidence = 1 - distance
                    # Only return results with high confidence (>60%)
                    if confidence > 0.6:
                        return {
                            'student_id': student_id,
                            'name': student['name'],
                            'email': student['email'],
                            'department': student['department'],
                            'year': student['year'],
                            'confidence': confidence
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error recognizing face from embedding: {e}")
            return None

    # Face Training System Methods
    
    def train_face_with_multiple_images(self, student_id: str, image_paths: List[str]) -> Dict[str, Any]:
        """Train face recognition with multiple images for better accuracy"""
        try:
            start_time = datetime.now()
            face_recognizer = get_face_recognizer()
            
            all_embeddings = []
            processed_images = 0
            successful_extractions = 0
            warnings = []
            
            logger.info(f"Starting face training for student {student_id} with {len(image_paths)} images")
            
            # Process each training image
            for i, image_path in enumerate(image_paths):
                try:
                    if not os.path.exists(image_path):
                        warnings.append(f"Image not found: {os.path.basename(image_path)}")
                        continue
                    
                    # Load and preprocess image
                    image = face_recognizer.load_image(image_path)
                    if image is None:
                        warnings.append(f"Could not load image: {os.path.basename(image_path)}")
                        continue
                    
                    processed_images += 1
                    
                    # Extract face embedding
                    embedding = face_recognizer.extract_face_embedding(image)
                    if embedding is not None:
                        all_embeddings.append(embedding)
                        successful_extractions += 1
                        logger.info(f"Extracted embedding from image {i+1}/{len(image_paths)}")
                    else:
                        warnings.append(f"No face detected in: {os.path.basename(image_path)}")
                        
                except Exception as e:
                    warnings.append(f"Error processing {os.path.basename(image_path)}: {str(e)}")
                    logger.error(f"Error processing image {image_path}: {e}")
            
            if len(all_embeddings) == 0:
                return {
                    'success': False,
                    'message': 'No valid face embeddings could be extracted from the provided images',
                    'processed_images': processed_images,
                    'successful_extractions': 0,
                    'warnings': warnings
                }
            
            # Create composite embedding (average of all embeddings for robustness)
            if len(all_embeddings) > 1:
                # Use weighted average with quality assessment
                composite_embedding = self._create_composite_embedding(all_embeddings)
            else:
                composite_embedding = all_embeddings[0]
            
            # Store the composite embedding
            self.face_embeddings[student_id] = composite_embedding
            self._save_face_embeddings()
            
            # Calculate training statistics
            processing_time = datetime.now() - start_time
            accuracy = (successful_extractions / len(image_paths)) * 100 if len(image_paths) > 0 else 0
            
            # Save training metadata
            self._save_training_metadata(student_id, {
                'training_date': datetime.now().isoformat(),
                'images_processed': processed_images,
                'successful_extractions': successful_extractions,
                'accuracy': accuracy,
                'processing_time': str(processing_time),
                'embedding_quality': self._assess_embedding_quality(all_embeddings)
            })
            
            logger.info(f"Face training completed for student {student_id}: {successful_extractions}/{processed_images} successful")
            
            return {
                'success': True,
                'message': f'Training completed successfully for {student_id}',
                'student_id': student_id,
                'processed_images': processed_images,
                'successful_extractions': successful_extractions,
                'accuracy': round(accuracy, 2),
                'processing_time': str(processing_time).split('.')[0],
                'embeddings_count': len(all_embeddings),
                'warnings': warnings,
                'confidence_score': round(self._calculate_confidence_score(all_embeddings), 2),
                'training_method': 'multi_image_composite'
            }
            
        except Exception as e:
            logger.error(f"Error training face for student {student_id}: {e}")
            return {
                'success': False,
                'message': f'Training failed: {str(e)}',
                'processed_images': 0,
                'successful_extractions': 0,
                'warnings': [str(e)]
            }
    
    def _create_composite_embedding(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Create a composite embedding from multiple embeddings"""
        try:
            # Simple average for now - could be enhanced with quality weighting
            composite = np.mean(embeddings, axis=0)
            
            # Normalize the composite embedding
            norm = np.linalg.norm(composite)
            if norm > 0:
                composite = composite / norm
            
            return composite
            
        except Exception as e:
            logger.error(f"Error creating composite embedding: {e}")
            # Fallback to first embedding
            return embeddings[0] if embeddings else None
    
    def _assess_embedding_quality(self, embeddings: List[np.ndarray]) -> float:
        """Assess the quality of embeddings based on consistency"""
        try:
            if len(embeddings) < 2:
                return 1.0
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Cosine similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(similarity)
            
            # Return average similarity as quality metric
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.error(f"Error assessing embedding quality: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, embeddings: List[np.ndarray]) -> float:
        """Calculate confidence score based on embedding consistency"""
        try:
            quality = self._assess_embedding_quality(embeddings)
            count_factor = min(len(embeddings) / 5.0, 1.0)  # Optimal around 5 images
            
            # Combine quality and count factors
            confidence = (quality * 0.7 + count_factor * 0.3) * 100
            return max(60.0, min(confidence, 95.0))  # Clamp between 60-95%
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 75.0
    
    def _save_training_metadata(self, student_id: str, metadata: Dict[str, Any]):
        """Save training metadata for tracking and analytics"""
        try:
            metadata_file = os.path.join(self.data_folder, 'training_metadata.json')
            
            # Load existing metadata
            training_data = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    training_data = json.load(f)
            
            # Update with new data
            training_data[student_id] = metadata
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(training_data, f, indent=2)
                
            logger.info(f"Training metadata saved for student {student_id}")
            
        except Exception as e:
            logger.error(f"Error saving training metadata: {e}")
    
    def get_training_metadata(self, student_id: str = None) -> Dict[str, Any]:
        """Get training metadata for a student or all students"""
        try:
            metadata_file = os.path.join(self.data_folder, 'training_metadata.json')
            
            if not os.path.exists(metadata_file):
                return {}
            
            with open(metadata_file, 'r') as f:
                training_data = json.load(f)
            
            if student_id:
                return training_data.get(student_id, {})
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error getting training metadata: {e}")
            return {}
    
    def get_trained_faces_summary(self) -> Dict[str, Any]:
        """Get summary of all trained faces"""
        try:
            students = self.get_all_students()
            training_metadata = self.get_training_metadata()
            
            trained_faces = []
            total_images = 0
            accuracy_scores = []
            
            for student in students:
                student_id = student['student_id']
                if student_id in self.face_embeddings:
                    metadata = training_metadata.get(student_id, {})
                    
                    images_count = metadata.get('successful_extractions', 1)
                    accuracy = metadata.get('accuracy', 75.0)
                    last_training = metadata.get('training_date', 'Unknown')
                    
                    # Format last training date
                    if last_training != 'Unknown':
                        try:
                            training_date = datetime.fromisoformat(last_training)
                            last_training = training_date.strftime('%Y-%m-%d')
                        except:
                            last_training = 'Unknown'
                    
                    trained_faces.append({
                        'student_id': student_id,
                        'name': student['name'],
                        'department': student.get('department', 'Unknown'),
                        'year': student.get('year', 'Unknown'),
                        'training_images_count': images_count,
                        'accuracy': round(accuracy, 1),
                        'last_training': last_training
                    })
                    
                    total_images += images_count
                    accuracy_scores.append(accuracy)
            
            average_accuracy = round(np.mean(accuracy_scores), 1) if accuracy_scores else 0
            
            return {
                'trained_faces': trained_faces,
                'statistics': {
                    'total_faces': len(trained_faces),
                    'total_images': total_images,
                    'average_accuracy': average_accuracy
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting trained faces summary: {e}")
            return {
                'trained_faces': [],
                'statistics': {
                    'total_faces': 0,
                    'total_images': 0,
                    'average_accuracy': 0
                }
            }
    
    def delete_face_training(self, student_id: str) -> bool:
        """Delete face training data for a student"""
        try:
            # Remove from embeddings
            if student_id in self.face_embeddings:
                del self.face_embeddings[student_id]
                self._save_face_embeddings()
            
            # Remove training metadata
            metadata_file = os.path.join(self.data_folder, 'training_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    training_data = json.load(f)
                
                if student_id in training_data:
                    del training_data[student_id]
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(training_data, f, indent=2)
            
            logger.info(f"Deleted face training data for student {student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting face training for student {student_id}: {e}")
            return False
    
    def export_training_data(self) -> Dict[str, Any]:
        """Export all training data for backup/analysis"""
        try:
            training_summary = self.get_trained_faces_summary()
            training_metadata = self.get_training_metadata()
            
            export_data = {
                'export_date': datetime.now().isoformat(),
                'system_version': '1.0',
                'trained_faces_summary': training_summary,
                'training_metadata': training_metadata,
                'embeddings_count': len(self.face_embeddings)
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return {}
    
    def bulk_mark_attendance_from_image(self, image_path: str, extract_location: bool = True) -> Dict[str, Any]:
        """Mark attendance for all recognized faces in an image with optional location extraction"""
        try:
            # Extract location data if requested
            location_data = None
            if extract_location:
                try:
                    from utils.geo_location import get_geo_location_extractor
                    geo_extractor = get_geo_location_extractor()
                    location_data = geo_extractor.extract_location_from_image(image_path)
                    logger.info(f"Location extraction {'successful' if location_data.get('has_location') else 'failed'} for image: {image_path}")
                except Exception as e:
                    logger.warning(f"Location extraction failed: {e}")
                    location_data = None
            
            recognition_result = self.recognize_face_from_image(image_path)

            if not recognition_result['success']:
                return {
                    'success': False,
                    'message': recognition_result.get('error', 'Failed to process image'),
                    'total_faces': 0,
                    'marked_attendance': [],
                    'already_marked': [],
                    'errors': [],
                    'location_data': location_data
                }

            recognized_faces = recognition_result['faces_recognized']

            results = {
                'success': True,
                'total_faces': recognition_result['faces_detected'],
                'faces_recognized': len(recognized_faces),
                'faces_unrecognized': len(recognition_result['faces_unrecognized']),
                'marked_attendance': [],
                'already_marked': [],
                'errors': [],
                'location_data': location_data
            }

            # Handle case where no faces were detected
            if recognition_result['faces_detected'] == 0:
                results['message'] = 'No faces detected in the image'
                return results

            # Handle case where faces were detected but none recognized
            if len(recognized_faces) == 0:
                results['message'] = f"Detected {recognition_result['faces_detected']} face(s) but none were recognized"
                return results

            for face in recognized_faces:
                student_id = face['student_id']
                confidence = face['confidence']

                # Mark attendance with location data
                success = self.mark_attendance(student_id, method="photo", confidence=confidence, location_data=location_data)

                if success:
                    attendance_record = {
                        'student_id': student_id,
                        'name': face['name'],
                        'confidence': confidence
                    }
                    
                    # Add location info to response if available
                    if location_data and location_data.get('has_location'):
                        attendance_record.update({
                            'location': location_data.get('address', 'Unknown Location'),
                            'coordinates': f"{location_data.get('latitude', 0):.6f}, {location_data.get('longitude', 0):.6f}"
                        })
                    
                    results['marked_attendance'].append(attendance_record)
                else:
                    # Check if already marked
                    today = datetime.now().strftime('%Y-%m-%d')
                    today_attendance = self.get_attendance_by_date(today)
                    already_marked = any(att['student_id'] == student_id for att in today_attendance)

                    if already_marked:
                        results['already_marked'].append({
                            'student_id': student_id,
                            'name': face['name']
                        })
                    else:
                        results['errors'].append({
                            'student_id': student_id,
                            'name': face['name'],
                            'error': 'Failed to mark attendance'
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error bulk marking attendance: {e}")
            return {
                'success': False,
                'total_faces': 0,
                'marked_attendance': [],
                'already_marked': [],
                'errors': [{'error': str(e)}],
                'location_data': None
            }
    
    def export_data(self, data_type: str) -> Optional[str]:
        """Export data as CSV file"""
        try:
            if data_type == 'students':
                return self.students_file
            elif data_type == 'attendance':
                return self.attendance_file
            else:
                return None
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None

# Global instance
data_manager = DataManager()

def get_data_manager():
    """Get the global data manager instance"""
    return data_manager
