import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from utils.security import SecureCSVHandler, validate_student_id, validate_email
try:
    from utils.face_recognition_utils import get_face_recognizer
except ImportError:
    # Fallback to mock for testing
    from utils.face_recognition_mock import get_face_recognizer
import pickle
import cv2
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """Secure data manager for student information and attendance"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.students_file = os.path.join(data_folder, "students.csv")
        self.attendance_file = os.path.join(data_folder, "attendance.csv")
        self.embeddings_file = os.path.join(data_folder, "face_embeddings.pkl")
        
        # CSV column definitions
        self.student_columns = ['student_id', 'name', 'email', 'department', 'year', 'registration_date']
        self.attendance_columns = ['student_id', 'name', 'date', 'time', 'status', 'confidence', 'method']
        
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
    
    def update_student(self, student_id: str, updated_data: Dict[str, Any]) -> bool:
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
            df = SecureCSVHandler.safe_read_csv(self.students_file)
            
            # Remove student from DataFrame
            df = df[df['student_id'] != student_id]
            
            # Save back to CSV
            SecureCSVHandler.safe_write_csv(df.to_dict('records'), self.students_file, self.student_columns)
            
            # Remove face embedding
            if student_id in self.face_embeddings:
                del self.face_embeddings[student_id]
                self._save_face_embeddings()
            
            logger.info(f"Deleted student: {student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting student {student_id}: {e}")
            return False
    
    def mark_attendance(self, student_id: str, method: str = "camera", confidence: float = 0.0) -> bool:
        """Mark attendance for a student"""
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
    
    def recognize_face_from_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Recognize faces from uploaded image"""
        try:
            face_recognizer = get_face_recognizer()
            image = face_recognizer.load_image(image_path)
            
            if image is None:
                return []
            
            # Extract all face embeddings from image
            face_embeddings = face_recognizer.extract_multiple_face_embeddings(image)
            face_locations = face_recognizer.detect_faces(image)
            
            recognized_faces = []
            
            for i, embedding in enumerate(face_embeddings):
                # Try to identify the face
                student_id, distance = face_recognizer.identify_face(embedding, self.face_embeddings)
                
                if student_id:
                    student = self.get_student(student_id)
                    if student:
                        confidence = 1 - distance  # Convert distance to confidence
                        location = face_locations[i] if i < len(face_locations) else None
                        
                        recognized_faces.append({
                            'student_id': student_id,
                            'name': student['name'],
                            'confidence': confidence,
                            'location': location
                        })
            
            return recognized_faces
            
        except Exception as e:
            logger.error(f"Error recognizing faces from image: {e}")
            return []
    
    def recognize_face_from_embedding(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Recognize face from embedding (for real-time recognition)"""
        try:
            face_recognizer = get_face_recognizer()
            student_id, distance = face_recognizer.identify_face(embedding, self.face_embeddings)
            
            if student_id:
                student = self.get_student(student_id)
                if student:
                    confidence = 1 - distance
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
    
    def bulk_mark_attendance_from_image(self, image_path: str) -> Dict[str, Any]:
        """Mark attendance for all recognized faces in an image"""
        try:
            recognized_faces = self.recognize_face_from_image(image_path)
            
            results = {
                'total_faces': len(recognized_faces),
                'marked_attendance': [],
                'already_marked': [],
                'errors': []
            }
            
            for face in recognized_faces:
                student_id = face['student_id']
                confidence = face['confidence']
                
                # Mark attendance
                success = self.mark_attendance(student_id, method="photo", confidence=confidence)
                
                if success:
                    results['marked_attendance'].append({
                        'student_id': student_id,
                        'name': face['name'],
                        'confidence': confidence
                    })
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
                'total_faces': 0,
                'marked_attendance': [],
                'already_marked': [],
                'errors': [{'error': str(e)}]
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
