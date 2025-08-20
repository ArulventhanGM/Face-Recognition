"""
Advanced Face Training Routes
Enhanced training endpoints with ML optimization
"""

from flask import jsonify, request, render_template
from flask_login import login_required
import os
import logging
from typing import List, Dict, Any
from utils.advanced_integration import TrainingManager, get_advanced_face_recognizer
from utils.data_manager import get_data_manager
import traceback

logger = logging.getLogger(__name__)

def register_advanced_training_routes(app):
    """Register advanced training routes with the Flask app"""
    
    @app.route('/advanced_training')
    @login_required
    def advanced_training():
        """Advanced training interface"""
        try:
            # Get current students
            data_manager = get_data_manager()
            students = data_manager.get_all_students()
            
            # Get training manager and status
            training_manager = TrainingManager()
            training_status = training_manager.get_training_status()
            
            # Get advanced face recognizer
            face_recognizer = get_advanced_face_recognizer()
            model_info = face_recognizer.get_model_info()
            
            return render_template('advanced_training.html',
                                 students=students,
                                 training_status=training_status,
                                 model_info=model_info)
                                 
        except Exception as e:
            logger.error(f"Error loading advanced training page: {e}")
            return render_template('advanced_training.html',
                                 error=f"Error loading training interface: {str(e)}",
                                 students=[],
                                 training_status={},
                                 model_info={})
    
    @app.route('/api/advanced_train_student', methods=['POST'])
    @login_required
    def advanced_train_student():
        """Train a specific student with advanced ML"""
        try:
            student_id = request.form.get('student_id')
            if not student_id:
                return jsonify({
                    'success': False,
                    'message': 'Student ID is required'
                })
            
            # Get uploaded files
            uploaded_files = request.files.getlist('training_images')
            
            if not uploaded_files or all(f.filename == '' for f in uploaded_files):
                return jsonify({
                    'success': False,
                    'message': 'No training images provided'
                })
            
            # Save uploaded files
            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            
            saved_paths = []
            for file in uploaded_files:
                if file and file.filename:
                    filename = f"training_{student_id}_{len(saved_paths)}_{file.filename}"
                    file_path = os.path.join(upload_folder, filename)
                    file.save(file_path)
                    saved_paths.append(file_path)
            
            if not saved_paths:
                return jsonify({
                    'success': False,
                    'message': 'No valid files were uploaded'
                })
            
            # Train using advanced system
            training_manager = TrainingManager()
            result = training_manager.add_training_images(student_id, saved_paths)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in advanced student training: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'message': f'Training failed: {str(e)}'
            })
    
    @app.route('/api/train_all_students_advanced', methods=['POST'])
    @login_required
    def train_all_students_advanced():
        """Train all students using the advanced ML system"""
        try:
            # Get all students
            data_manager = get_data_manager()
            students = data_manager.get_all_students()
            
            if not students:
                return jsonify({
                    'success': False,
                    'message': 'No students found to train'
                })
            
            # Train all students
            training_manager = TrainingManager()
            result = training_manager.train_all_students(students)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error training all students: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'message': f'Batch training failed: {str(e)}'
            })
    
    @app.route('/api/optimize_model', methods=['POST'])
    @login_required
    def optimize_model():
        """Optimize the ML model with cross-validation"""
        try:
            face_recognizer = get_advanced_face_recognizer()
            result = face_recognizer.optimize_model()
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'message': f'Model optimization failed: {str(e)}'
            })
    
    @app.route('/api/get_training_status')
    @login_required
    def get_training_status():
        """Get comprehensive training status"""
        try:
            training_manager = TrainingManager()
            status = training_manager.get_training_status()
            
            # Add validation results
            validation = training_manager.validate_model_performance()
            status['validation'] = validation
            
            return jsonify({
                'success': True,
                'status': status
            })
            
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return jsonify({
                'success': False,
                'message': f'Failed to get training status: {str(e)}'
            })
    
    @app.route('/api/delete_training_data', methods=['POST'])
    @login_required
    def delete_training_data():
        """Delete training data for a student"""
        try:
            data = request.get_json()
            student_id = data.get('student_id')
            
            if not student_id:
                return jsonify({
                    'success': False,
                    'message': 'Student ID is required'
                })
            
            face_recognizer = get_advanced_face_recognizer()
            result = face_recognizer.delete_face_training(student_id)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error deleting training data: {e}")
            return jsonify({
                'success': False,
                'message': f'Failed to delete training data: {str(e)}'
            })
    
    @app.route('/api/retrain_model', methods=['POST'])
    @login_required
    def retrain_model():
        """Retrain the entire model"""
        try:
            face_recognizer = get_advanced_face_recognizer()
            result = face_recognizer.retrain_model()
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return jsonify({
                'success': False,
                'message': f'Model retraining failed: {str(e)}'
            })
    
    @app.route('/api/test_advanced_recognition', methods=['POST'])
    @login_required
    def test_advanced_recognition():
        """Test recognition with uploaded image"""
        try:
            if 'test_image' not in request.files:
                return jsonify({
                    'success': False,
                    'message': 'No test image provided'
                })
            
            file = request.files['test_image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'message': 'No file selected'
                })
            
            # Save test image
            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            
            test_path = os.path.join(upload_folder, f"test_{file.filename}")
            file.save(test_path)
            
            # Load image and test recognition
            import cv2
            image = cv2.imread(test_path)
            
            if image is None:
                return jsonify({
                    'success': False,
                    'message': 'Could not load image'
                })
            
            # Test with advanced system
            face_recognizer = get_advanced_face_recognizer()
            results = face_recognizer.recognize_face(image, min_confidence=0.5)
            
            # Clean up test image
            try:
                os.remove(test_path)
            except:
                pass
            
            return jsonify({
                'success': True,
                'results': results,
                'faces_detected': len(results)
            })
            
        except Exception as e:
            logger.error(f"Error testing recognition: {e}")
            return jsonify({
                'success': False,
                'message': f'Recognition test failed: {str(e)}'
            })
