from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, abort
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import os
import cv2
import numpy as np
import base64
import pandas as pd
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter

from config import Config
from utils.data_manager import get_data_manager, get_face_recognizer
from utils.security import sanitize_filename, validate_student_id, validate_email
from utils.data_manager import get_face_recognition_backend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'error'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# Simple User class for admin authentication
class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    if user_id == app.config['ADMIN_USERNAME']:
        return User(user_id)
    return None

# Helper functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Save uploaded file and return the path"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = sanitize_filename(filename)
        
        # Add timestamp to prevent conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

def decode_base64_image(image_data):
    """Decode base64 image data to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None

def get_dashboard_stats():
    """Get statistics for dashboard"""
    data_manager = get_data_manager()
    
    # Get all students and attendance
    students = data_manager.get_all_students()
    all_attendance = data_manager.get_all_attendance()
    
    # Calculate stats
    total_students = len(students)
    
    today = datetime.now().strftime('%Y-%m-%d')
    today_attendance = [att for att in all_attendance if att['date'] == today]
    total_attendance_today = len(today_attendance)
    
    # This week
    week_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    week_attendance = [att for att in all_attendance if att['date'] >= week_start]
    total_attendance_week = len(week_attendance)
    
    # This month
    month_start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
    month_attendance = [att for att in all_attendance if att['date'] >= month_start]
    total_attendance_month = len(month_attendance)
    
    return {
        'total_students': total_students,
        'total_attendance_today': total_attendance_today,
        'total_attendance_week': total_attendance_week,
        'total_attendance_month': total_attendance_month
    }

# Routes
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring and load balancing"""
    try:
        # Basic system checks
        data_manager = get_data_manager()
        
        # Check if core components are working
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'data_manager': 'ok',
                'face_recognition': 'ok',
                'upload_folder': 'ok',
                'data_folder': 'ok'
            },
            'system_info': {
                'face_recognition_backend': get_face_recognition_backend(),
                'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
                'data_folder_exists': os.path.exists(app.config['DATA_FOLDER'])
            }
        }
        
        # Check folders
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            health_status['components']['upload_folder'] = 'missing'
        if not os.path.exists(app.config['DATA_FOLDER']):
            health_status['components']['data_folder'] = 'missing'
            
        # Check if any component failed
        failed_components = [k for k, v in health_status['components'].items() if v != 'ok']
        if failed_components:
            health_status['status'] = 'degraded'
            health_status['failed_components'] = failed_components
        
        return jsonify(health_status), 200 if health_status['status'] == 'healthy' else 503
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 503

@app.route('/')
def index():
    """Redirect to dashboard or login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if (username == app.config['ADMIN_USERNAME'] and 
            password == app.config['ADMIN_PASSWORD']):
            user = User(username)
            login_user(user)
            flash('Login successful!', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    """Admin logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Admin dashboard"""
    stats = get_dashboard_stats()
    
    # Get recent attendance (last 10 records)
    data_manager = get_data_manager()
    all_attendance = data_manager.get_all_attendance()
    recent_attendance = sorted(all_attendance, 
                             key=lambda x: f"{x['date']} {x['time']}", 
                             reverse=True)[:10]
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         recent_attendance=recent_attendance)

@app.route('/diagnostics')
@login_required
def diagnostics():
    """Runtime diagnostics for face recognition backend & data health."""
    dm = get_data_manager()
    backend = get_face_recognition_backend() if 'get_face_recognition_backend' in globals() else 'unknown'
    return jsonify({
        'backend': backend,
        'students_count': len(dm.get_all_students()),
        'embeddings_count': len(dm.face_embeddings),
        'attendance_records': len(dm.get_all_attendance())
    })

@app.route('/students')
@login_required
def students():
    """List all students"""
    data_manager = get_data_manager()
    students = data_manager.get_all_students()
    
    # Get face data availability
    face_data = data_manager.face_embeddings
    students_with_face_data = len([s for s in students if s['student_id'] in face_data])
    
    # Group by department and year
    students_by_department = Counter([s['department'] for s in students])
    students_by_year = Counter([s['year'] for s in students])
    
    return render_template('students.html', 
                         students=students,
                         face_data=face_data,
                         students_with_face_data=students_with_face_data,
                         students_by_department=students_by_department,
                         students_by_year=students_by_year)

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    """Add new student"""
    if request.method == 'POST':
        try:
            # Get form data
            student_data = {
                'student_id': request.form.get('student_id', '').strip(),
                'name': request.form.get('name', '').strip(),
                'email': request.form.get('email', '').strip(),
                'department': request.form.get('department', '').strip(),
                'year': request.form.get('year', '').strip()
            }
            
            # Enhanced validation with specific error messages
            if not student_data['student_id']:
                flash('Student ID is required', 'error')
                return render_template('add_student.html')
            
            if not student_data['name']:
                flash('Student name is required', 'error')
                return render_template('add_student.html')
            
            if not student_data['email']:
                flash('Email address is required', 'error')
                return render_template('add_student.html')
            
            if not student_data['department']:
                flash('Department selection is required', 'error')
                return render_template('add_student.html')
            
            if not student_data['year']:
                flash('Academic year selection is required', 'error')
                return render_template('add_student.html')
            
            # Validate student ID format
            if not validate_student_id(student_data['student_id']):
                flash('Invalid student ID format. Use alphanumeric characters (3-20 chars)', 'error')
                return render_template('add_student.html')
            
            # Validate email format
            if not validate_email(student_data['email']):
                flash('Invalid email format. Please enter a valid email address', 'error')
                return render_template('add_student.html')
            
            # Check for existing student
            data_manager = get_data_manager()
            existing_student = data_manager.get_student(student_data['student_id'])
            if existing_student:
                flash(f'Student ID {student_data["student_id"]} already exists', 'error')
                return render_template('add_student.html')
            
            # Handle face image upload
            face_image_path = None
            upload_success = True
            
            # Check if face image is required (for new students)
            if 'face_image' in request.files:
                file = request.files['face_image']
                if file and file.filename:
                    # Validate file
                    if not allowed_file(file.filename):
                        flash('Invalid image format. Supported formats: PNG, JPG, JPEG, GIF, BMP', 'error')
                        return render_template('add_student.html')
                    
                    # Check file size (additional check beyond Flask's MAX_CONTENT_LENGTH)
                    file.seek(0, os.SEEK_END)
                    file_size = file.tell()
                    file.seek(0)
                    
                    if file_size > 16 * 1024 * 1024:  # 16MB
                        flash('Image file too large. Maximum size is 16MB', 'error')
                        return render_template('add_student.html')
                    
                    face_image_path = save_uploaded_file(file)
                    if not face_image_path:
                        flash('Failed to save uploaded image. Please try again', 'error')
                        upload_success = False
                else:
                    # No file uploaded but face_image field exists
                    flash('Face photo is required for new student registration', 'error')
                    return render_template('add_student.html')
            else:
                # face_image field not in request at all
                flash('Face photo is required for new student registration', 'error')
                return render_template('add_student.html')
            
            # Add student to database
            if upload_success:
                success = data_manager.add_student(student_data, face_image_path)
                
                if success:
                    success_msg = f'Student {student_data["name"]} (ID: {student_data["student_id"]}) added successfully!'
                    if face_image_path:
                        success_msg += ' Face photo uploaded and processed.'
                    flash(success_msg, 'success')
                    return redirect(url_for('students'))
                else:
                    flash('Failed to add student to database. Please check the data and try again', 'error')
            
        except Exception as e:
            logger.error(f"Error adding student: {e}")
            flash('An unexpected error occurred while adding the student. Please try again', 'error')
    
    return render_template('add_student.html')

@app.route('/edit_student/<student_id>', methods=['GET', 'POST'])
@login_required
def edit_student(student_id):
    """Edit existing student"""
    data_manager = get_data_manager()
    student = data_manager.get_student(student_id)
    
    if not student:
        flash('Student not found', 'error')
        return redirect(url_for('students'))
    
    if request.method == 'POST':
        try:
            # Get updated data
            updated_data = {
                'name': request.form.get('name', '').strip(),
                'email': request.form.get('email', '').strip(),
                'department': request.form.get('department', '').strip(),
                'year': request.form.get('year', '').strip()
            }
            
            # Validate required fields
            required_fields = ['name', 'email', 'department', 'year']
            missing_fields = [field for field in required_fields if not updated_data.get(field)]

            if missing_fields:
                flash(f'The following fields are required: {", ".join(missing_fields)}', 'error')
                return render_template('add_student.html', student=student)
            
            if not validate_email(updated_data['email']):
                flash('Invalid email format', 'error')
                return render_template('add_student.html', student=student)
            
            # Handle face image update
            new_photo_path = None
            if 'face_image' in request.files:
                file = request.files['face_image']
                if file and file.filename:
                    face_image_path = save_uploaded_file(file)
                    if face_image_path:
                        # Extract new face embedding
                        face_recognizer = get_face_recognizer()
                        image = face_recognizer.load_image(face_image_path)
                        if image is not None:
                            embedding = face_recognizer.extract_face_embedding(image)
                            if embedding is not None:
                                data_manager.face_embeddings[student_id] = embedding
                                data_manager._save_face_embeddings()
                                new_photo_path = face_image_path  # Store the new photo path
                                flash('Face data and photo updated successfully!', 'success')
                            else:
                                flash('Could not extract face from image', 'warning')
                                # Still save the photo even if face extraction failed
                                new_photo_path = face_image_path
                        else:
                            flash('Could not load image', 'warning')
                            # Still save the photo path
                            new_photo_path = face_image_path

            # Update student data
            success = data_manager.update_student(student_id, updated_data, new_photo_path)
            
            if success:
                flash(f'Student {updated_data["name"]} updated successfully!', 'success')
                return redirect(url_for('students'))
            else:
                flash('Failed to update student', 'error')
                
        except Exception as e:
            logger.error(f"Error updating student: {e}")
            flash('An error occurred while updating the student', 'error')
    
    # Get additional data for template
    has_face_data = student_id in data_manager.face_embeddings
    
    # Get recent attendance
    all_attendance = data_manager.get_all_attendance()
    recent_attendance = [att for att in all_attendance if att['student_id'] == student_id]
    recent_attendance = sorted(recent_attendance, 
                             key=lambda x: f"{x['date']} {x['time']}", 
                             reverse=True)[:5]
    
    return render_template('add_student.html', 
                         student=student,
                         has_face_data=has_face_data,
                         recent_attendance=recent_attendance)

@app.route('/delete_student/<student_id>')
@login_required
def delete_student(student_id):
    """Delete student"""
    data_manager = get_data_manager()
    student = data_manager.get_student(student_id)
    
    if not student:
        flash('Student not found', 'error')
        return redirect(url_for('students'))
    
    success = data_manager.delete_student(student_id)
    
    if success:
        flash(f'Student {student["name"]} deleted successfully', 'success')
    else:
        flash('Failed to delete student', 'error')
    
    return redirect(url_for('students'))

@app.route('/recognition')
@login_required
def recognition():
    """Face recognition page"""
    return render_template('recognition.html')

@app.route('/recognize_photo', methods=['POST'])
@login_required
def recognize_photo():
    """Recognize faces from uploaded photo"""
    try:
        if 'photo' not in request.files:
            return jsonify({'success': False, 'message': 'No photo uploaded'})

        file = request.files['photo']
        if not file or not file.filename:
            return jsonify({'success': False, 'message': 'No photo selected'})

        # Save uploaded file
        filepath = save_uploaded_file(file)
        if not filepath:
            return jsonify({'success': False, 'message': 'Invalid image file'})

        try:
            # Recognize faces
            data_manager = get_data_manager()
            recognition_result = data_manager.recognize_face_from_image(filepath)

            # Debug logging
            logger.info(f"Recognition result: {recognition_result}")

            if not recognition_result['success']:
                return jsonify({
                    'success': False,
                    'message': recognition_result.get('error', 'Failed to process image')
                })

            # Return comprehensive results
            return jsonify({
                'success': True,
                'data': recognition_result['faces_recognized'],
                'faces_detected': recognition_result['faces_detected'],
                'faces_recognized': len(recognition_result['faces_recognized']),
                'faces_unrecognized': len(recognition_result['faces_unrecognized']),
                'unrecognized_details': recognition_result['faces_unrecognized'],
                'message': recognition_result.get('message', f'Processed {recognition_result["faces_detected"]} face(s)')
            })

        finally:
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass

    except Exception as e:
        logger.error(f"Error recognizing photo: {e}")
        return jsonify({'success': False, 'message': 'Failed to process photo'})

@app.route('/process_photo', methods=['POST'])
@login_required
def process_photo():
    """Process captured photo from camera for recognition"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        image_data = data['image']
        
        # Decode base64 image
        image = decode_base64_image(image_data)
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image data'})
        
        # Save temporary image file
        temp_filename = f"temp_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        cv2.imwrite(temp_filepath, image)
        
        try:
            # Use face recognizer to process image
            face_recognizer = get_face_recognizer()
            results = face_recognizer.recognize_faces_from_image(temp_filepath)
            
            return jsonify({
                'success': True,
                'data': results,
                'message': f'Recognized {len(results)} face(s)'
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                
    except Exception as e:
        logger.error(f"Error processing camera photo: {e}")
        return jsonify({'success': False, 'message': 'Failed to process camera photo'})

@app.route('/recognize_realtime', methods=['POST'])
@login_required
def recognize_realtime():
    """Enhanced real-time face recognition from camera"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data'})

        # Decode base64 image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image data'})

        # Get face recognizer and data manager
        face_recognizer = get_face_recognizer()
        data_manager = get_data_manager()

        # Process frame for real-time recognition
        face_locations, face_embeddings = face_recognizer.process_webcam_frame(image)

        recognized_faces = []

        for embedding in face_embeddings:
            result = data_manager.recognize_face_from_embedding(embedding)
            if result:
                recognized_faces.append(result)
        
        return jsonify({
            'success': True,
            'data': recognized_faces,
            'face_locations': face_locations,
            'faces_detected': len(face_locations),
            'faces_recognized': len(recognized_faces)
        })
        
    except Exception as e:
        logger.error(f"Error in real-time recognition: {e}")
        return jsonify({'success': False, 'message': 'Recognition failed'})

@app.route('/mark_attendance', methods=['POST'])
@login_required
def mark_attendance():
    """Mark attendance for a student"""
    try:
        data = request.get_json()
        if not data or 'student_id' not in data:
            return jsonify({'success': False, 'message': 'Student ID required'})
        
        student_id = data['student_id']
        data_manager = get_data_manager()
        
        success = data_manager.mark_attendance(student_id, method='camera')
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Attendance marked for {student_id}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Attendance already marked or student not found'
            })
            
    except Exception as e:
        logger.error(f"Error marking attendance: {e}")
        return jsonify({'success': False, 'message': 'Failed to mark attendance'})

@app.route('/bulk_attendance', methods=['POST'])
@login_required
def bulk_attendance():
    """Mark attendance from group photo with optional location extraction"""
    try:
        if 'photo' not in request.files:
            return jsonify({'success': False, 'message': 'No photo uploaded'})
        
        file = request.files['photo']
        if not file or not file.filename:
            return jsonify({'success': False, 'message': 'No photo selected'})
        
        # Check if location extraction is requested (default: True)
        extract_location = request.form.get('extract_location', 'true').lower() == 'true'
        
        # Save uploaded file
        filepath = save_uploaded_file(file)
        if not filepath:
            return jsonify({'success': False, 'message': 'Invalid image file'})
        
        try:
            # Process group photo for attendance with location extraction
            data_manager = get_data_manager()
            results = data_manager.bulk_mark_attendance_from_image(filepath, extract_location=extract_location)
            
            # Add location information to response message if available
            message = 'Bulk attendance processing completed'
            if results.get('location_data') and results['location_data'].get('has_location'):
                location = results['location_data'].get('address', 'Unknown Location')
                message += f' with location: {location}'
            elif extract_location:
                message += ' (no location data found in image)'
            
            return jsonify({
                'success': True,
                'data': results,
                'message': message
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        logger.error(f"Error processing group photo: {e}")
        return jsonify({'success': False, 'message': 'Failed to process group photo'})

@app.route('/mark_bulk_attendance', methods=['POST'])
@login_required
def mark_bulk_attendance():
    """Mark attendance for multiple students (from real-time recognition)"""
    try:
        data = request.get_json()
        if not data or 'student_ids' not in data:
            return jsonify({'success': False, 'message': 'Student IDs required'})

        student_ids = data['student_ids']
        method = data.get('method', 'bulk')

        data_manager = get_data_manager()
        marked_count = 0
        errors = []

        for student_id in student_ids:
            try:
                success = data_manager.mark_attendance(student_id, method=method)
                if success:
                    marked_count += 1
                else:
                    errors.append(f'Failed to mark attendance for {student_id}')
            except Exception as e:
                errors.append(f'Error marking attendance for {student_id}: {str(e)}')

        return jsonify({
            'success': True,
            'marked_count': marked_count,
            'total_requested': len(student_ids),
            'errors': errors
        })

    except Exception as e:
        logger.error(f"Error marking bulk attendance: {e}")
        return jsonify({'success': False, 'message': 'Failed to mark bulk attendance'})

@app.route('/test_photo/<student_id>')
def test_photo(student_id):
    """Test photo endpoint without authentication"""
    return f"Test photo for student {student_id}"

@app.route('/student_photo/<student_id>')
def student_photo(student_id):
    """Serve student photo with enhanced error handling"""
    print(f"DEBUG: student_photo called with ID: {student_id}")
    logger.info(f"Student photo request for ID: {student_id}")
    try:
        data_manager = get_data_manager()
        student = data_manager.get_student(student_id)

        if not student:
            logger.warning(f"Student not found: {student_id}")
            # Serve default avatar directly instead of redirect
            default_avatar_path = os.path.join(app.static_folder, 'images', 'default-avatar.svg')
            if os.path.exists(default_avatar_path):
                logger.info(f"Serving default avatar for non-existent student {student_id}")
                return send_file(default_avatar_path, mimetype='image/svg+xml')
            else:
                logger.error(f"Default avatar not found at {default_avatar_path}")
                abort(404)

        # Check if student has a stored photo path
        photo_path = student.get('photo_path', '')
        logger.debug(f"Student {student_id} photo_path: {repr(photo_path)}")
        
        if photo_path and pd.notna(photo_path) and str(photo_path).strip():
            # Convert to string and handle relative path
            photo_path_str = str(photo_path).strip()
            if not os.path.isabs(photo_path_str):
                photo_path_str = os.path.join(os.getcwd(), photo_path_str)
            
            logger.debug(f"Checking photo path for {student_id}: {photo_path_str}")
            
            if os.path.exists(photo_path_str):
                logger.info(f"Serving photo for {student_id}: {photo_path_str}")
                return send_file(photo_path_str)
            else:
                logger.warning(f"Photo not found at path: {photo_path_str}")

        # Fallback 1: Look in data/photos directory
        photos_folder = os.path.join(app.config['DATA_FOLDER'], 'photos')
        os.makedirs(photos_folder, exist_ok=True)
        logger.debug(f"Checking photos folder: {photos_folder}")
        
        for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            photo_file = os.path.join(photos_folder, f"{student_id}.{ext}")
            if os.path.exists(photo_file):
                logger.info(f"Found photo in photos folder: {photo_file}")
                return send_file(photo_file)

        # Fallback 2: Look for student photo in uploads folder
        upload_folder = app.config['UPLOAD_FOLDER']
        logger.debug(f"Checking uploads folder: {upload_folder}")
        if os.path.exists(upload_folder):
            # Check for common image extensions with student ID
            for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                fallback_path = os.path.join(upload_folder, f"{student_id}.{ext}")
                if os.path.exists(fallback_path):
                    logger.info(f"Found photo in uploads: {fallback_path}")
                    return send_file(fallback_path)

            # Look for timestamped files containing student_id
            try:
                for filename in os.listdir(upload_folder):
                    if (student_id.lower() in filename.lower() and 
                        any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'])):
                        fallback_path = os.path.join(upload_folder, filename)
                        if os.path.exists(fallback_path):
                            logger.info(f"Found timestamped photo: {fallback_path}")
                            return send_file(fallback_path)
            except OSError as e:
                logger.error(f"Error listing uploads directory: {e}")

        # No photo found - serve default avatar directly
        default_avatar_path = os.path.join(app.static_folder, 'images', 'default-avatar.svg')
        if os.path.exists(default_avatar_path):
            logger.info(f"No photo found for student {student_id}, serving default avatar")
            return send_file(default_avatar_path, mimetype='image/svg+xml')
        else:
            # If default avatar doesn't exist, return a 404
            logger.error(f"Default avatar not found at {default_avatar_path}")
            abort(404)

    except Exception as e:
        logger.error(f"Error serving student photo for {student_id}: {e}")
        # Try to serve default avatar on error too
        try:
            default_avatar_path = os.path.join(app.static_folder, 'images', 'default-avatar.svg')
            if os.path.exists(default_avatar_path):
                return send_file(default_avatar_path, mimetype='image/svg+xml')
            else:
                abort(404)
        except:
            abort(404)

@app.route('/attendance')
@login_required
def attendance():
    """Attendance management page"""
    data_manager = get_data_manager()
    
    # Get all attendance records
    all_attendance = data_manager.get_all_attendance()
    
    # Get all students for department info
    all_students = data_manager.get_all_students()
    student_info = {s['student_id']: s for s in all_students}
    
    # Add student info to attendance records
    for record in all_attendance:
        if record['student_id'] in student_info:
            student = student_info[record['student_id']]
            record['department'] = student.get('department', 'N/A')
            record['year'] = student.get('year', 'N/A')
    
    # Apply filters
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    student_id_filter = request.args.get('student_id')
    department_filter = request.args.get('department')
    method_filter = request.args.get('method')
    
    filtered_attendance = all_attendance
    
    if date_from:
        filtered_attendance = [att for att in filtered_attendance if att['date'] >= date_from]
    
    if date_to:
        filtered_attendance = [att for att in filtered_attendance if att['date'] <= date_to]
    
    if student_id_filter:
        filtered_attendance = [att for att in filtered_attendance 
                             if student_id_filter.lower() in att['student_id'].lower()]
    
    if department_filter:
        filtered_attendance = [att for att in filtered_attendance 
                             if att.get('department') == department_filter]
    
    if method_filter:
        filtered_attendance = [att for att in filtered_attendance 
                             if att['method'] == method_filter]
    
    # Sort by date and time (newest first)
    filtered_attendance = sorted(filtered_attendance, 
                               key=lambda x: f"{x['date']} {x['time']}", 
                               reverse=True)
    
    # Limit to 100 records for performance
    filtered_attendance = filtered_attendance[:100]
    
    # Calculate statistics
    today = datetime.now().strftime('%Y-%m-%d')
    week_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    month_start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
    
    stats = {
        'total_today': len([att for att in all_attendance if att['date'] == today]),
        'total_week': len([att for att in all_attendance if att['date'] >= week_start]),
        'total_month': len([att for att in all_attendance if att['date'] >= month_start]),
        'avg_daily': len(all_attendance) // max(1, len(set([att['date'] for att in all_attendance])))
    }
    
    # Get unique departments for filter
    departments = sorted(set([att.get('department', '') for att in all_attendance if att.get('department')]))
    
    # Attendance insights
    # Last 7 days attendance
    last_7_days = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    attendance_by_date = {}
    for date in last_7_days:
        count = len([att for att in all_attendance if att['date'] == date])
        attendance_by_date[date] = count
    
    # Department-wise attendance
    attendance_by_department = Counter([att.get('department', 'Unknown') for att in all_attendance])
    
    return render_template('attendance.html',
                         attendance_records=filtered_attendance,
                         stats=stats,
                         departments=departments,
                         attendance_by_date=attendance_by_date,
                         attendance_by_department=attendance_by_department)

@app.route('/attendance_today')
@login_required
def attendance_today():
    """Today's attendance"""
    today = datetime.now().strftime('%Y-%m-%d')
    return redirect(url_for('attendance', date_from=today, date_to=today))

@app.route('/reports')
@login_required
def reports():
    """Reports page"""
    # This could be expanded with more detailed reporting
    return redirect(url_for('attendance'))

@app.route('/export_data/<data_type>')
@login_required
def export_data(data_type):
    """Export data as CSV"""
    try:
        data_manager = get_data_manager()
        file_path = data_manager.export_data(data_type)
        
        if file_path and os.path.exists(file_path):
            return send_file(file_path, 
                           as_attachment=True, 
                           download_name=f'{data_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                           mimetype='text/csv')
        else:
            flash(f'No {data_type} data available for export', 'error')
            
    except Exception as e:
        logger.error(f"Error exporting {data_type}: {e}")
        flash(f'Failed to export {data_type} data', 'error')
    
    return redirect(url_for('dashboard'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500

# Face Training System Routes

@app.route('/face_training')
@login_required
def face_training():
    """Face training interface"""
    try:
        data_manager = get_data_manager()
        students = data_manager.get_all_students()
        
        # Get training statistics
        training_summary = data_manager.get_trained_faces_summary()
        
        return render_template('face_training.html',
                             students=students,
                             trained_faces_count=training_summary['statistics']['total_faces'],
                             total_training_images=training_summary['statistics']['total_images'],
                             average_accuracy=training_summary['statistics']['average_accuracy'])
    except Exception as e:
        logger.error(f"Error loading face training page: {e}")
        flash('Error loading face training interface', 'error')
        return redirect(url_for('dashboard'))

@app.route('/train_existing_student', methods=['POST'])
@login_required
def train_existing_student():
    """Train face recognition for an existing student"""
    try:
        student_id = request.form.get('student_id')
        if not student_id:
            return jsonify({'success': False, 'message': 'Student ID is required'})
        
        data_manager = get_data_manager()
        student = data_manager.get_student(student_id)
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'})
        
        # Handle uploaded training images
        training_images = request.files.getlist('training_images')
        if len(training_images) < 3:
            return jsonify({'success': False, 'message': 'At least 3 training images are required'})
        
        # Save uploaded images temporarily
        temp_image_paths = []
        for i, image_file in enumerate(training_images):
            if image_file and allowed_file(image_file.filename):
                filename = secure_filename(f"{student_id}_training_{i}_{image_file.filename}")
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(temp_path)
                temp_image_paths.append(temp_path)
        
        if len(temp_image_paths) == 0:
            return jsonify({'success': False, 'message': 'No valid images uploaded'})
        
        # Perform training
        result = data_manager.train_face_with_multiple_images(student_id, temp_image_paths)
        
        # Clean up temporary files
        for temp_path in temp_image_paths:
            try:
                os.remove(temp_path)
            except:
                pass
        
        # Add student information to result
        if result['success']:
            result['student_name'] = student['name']
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error training existing student: {e}")
        return jsonify({'success': False, 'message': f'Training failed: {str(e)}'})

@app.route('/train_new_person', methods=['POST'])
@login_required
def train_new_person():
    """Register a new person and train face recognition simultaneously"""
    try:
        # Get form data
        student_data = {
            'student_id': request.form.get('student_id', '').strip(),
            'name': request.form.get('name', '').strip(),
            'email': request.form.get('email', '').strip(),
            'department': request.form.get('department', '').strip(),
            'year': request.form.get('year', '').strip()
        }
        
        # Validate required fields
        for field, value in student_data.items():
            if not value:
                return jsonify({'success': False, 'message': f'{field.replace("_", " ").title()} is required'})
        
        # Handle uploaded training images
        training_images = request.files.getlist('training_images')
        if len(training_images) < 3:
            return jsonify({'success': False, 'message': 'At least 3 training images are required'})
        
        data_manager = get_data_manager()
        
        # Check if student already exists
        if data_manager.get_student(student_data['student_id']):
            return jsonify({'success': False, 'message': 'Student ID already exists'})
        
        # Save uploaded images temporarily
        temp_image_paths = []
        for i, image_file in enumerate(training_images):
            if image_file and allowed_file(image_file.filename):
                filename = secure_filename(f"{student_data['student_id']}_training_{i}_{image_file.filename}")
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(temp_path)
                temp_image_paths.append(temp_path)
        
        if len(temp_image_paths) == 0:
            return jsonify({'success': False, 'message': 'No valid images uploaded'})
        
        # Register the student first (without face data)
        student_added = data_manager.add_student(student_data)
        if not student_added:
            return jsonify({'success': False, 'message': 'Failed to register student'})
        
        # Perform face training
        result = data_manager.train_face_with_multiple_images(student_data['student_id'], temp_image_paths)
        
        # Clean up temporary files
        for temp_path in temp_image_paths:
            try:
                os.remove(temp_path)
            except:
                pass
        
        # Add student information to result
        if result['success']:
            result['student_name'] = student_data['name']
            
            # Save the first image as the student's profile photo
            if len(temp_image_paths) > 0:
                try:
                    profile_filename = secure_filename(f"{student_data['student_id']}.jpg")
                    profile_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_filename)
                    # Re-save the first training image as profile photo
                    first_image = training_images[0]
                    first_image.stream.seek(0)  # Reset stream position
                    first_image.save(profile_path)
                    
                    # Update student with photo path
                    data_manager.update_student(student_data['student_id'], {'photo_path': profile_path})
                except Exception as e:
                    logger.warning(f"Could not save profile photo: {e}")
        else:
            # If face training failed, remove the student
            data_manager.delete_student(student_data['student_id'])
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error training new person: {e}")
        return jsonify({'success': False, 'message': f'Registration and training failed: {str(e)}'})

@app.route('/get_trained_faces')
@login_required
def get_trained_faces():
    """Get list of all trained faces with statistics"""
    try:
        data_manager = get_data_manager()
        training_summary = data_manager.get_trained_faces_summary()
        return jsonify(training_summary)
        
    except Exception as e:
        logger.error(f"Error getting trained faces: {e}")
        return jsonify({
            'trained_faces': [],
            'statistics': {
                'total_faces': 0,
                'total_images': 0,
                'average_accuracy': 0
            }
        })

@app.route('/test_face_recognition/<student_id>')
@login_required
def test_face_recognition(student_id):
    """Test face recognition for a specific student"""
    try:
        data_manager = get_data_manager()
        
        # Check if student has face training data
        if student_id not in data_manager.face_embeddings:
            return jsonify({'success': False, 'message': 'No face training data found'})
        
        # Get student info
        student = data_manager.get_student(student_id)
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'})
        
        # Get training metadata
        metadata = data_manager.get_training_metadata(student_id)
        confidence = metadata.get('accuracy', 75.0)
        
        return jsonify({
            'success': True,
            'student_id': student_id,
            'student_name': student['name'],
            'confidence': round(confidence, 2),
            'training_date': metadata.get('training_date', 'Unknown'),
            'images_processed': metadata.get('successful_extractions', 1)
        })
        
    except Exception as e:
        logger.error(f"Error testing face recognition for {student_id}: {e}")
        return jsonify({'success': False, 'message': f'Test failed: {str(e)}'})

@app.route('/delete_face_training/<student_id>', methods=['DELETE'])
@login_required
def delete_face_training(student_id):
    """Delete face training data for a student"""
    try:
        data_manager = get_data_manager()
        
        # Check if student exists
        student = data_manager.get_student(student_id)
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'})
        
        # Delete face training data
        success = data_manager.delete_face_training(student_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Face training data deleted for {student["name"]}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to delete face training data'
            })
        
    except Exception as e:
        logger.error(f"Error deleting face training for {student_id}: {e}")
        return jsonify({'success': False, 'message': f'Delete failed: {str(e)}'})

@app.route('/export_training_data')
@login_required
def export_training_data():
    """Export training data as JSON"""
    try:
        data_manager = get_data_manager()
        export_data = data_manager.export_training_data()
        
        # Create JSON response
        response = jsonify(export_data)
        response.headers['Content-Disposition'] = f'attachment; filename=training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        response.headers['Content-Type'] = 'application/json'
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting training data: {e}")
        return jsonify({'error': 'Export failed'}), 500

@app.route('/batch_face_training', methods=['POST'])
@login_required
def batch_face_training():
    """Handle batch face training with multiple images per student"""
    try:
        student_id = request.form.get('student_id')
        if not student_id:
            return jsonify({'success': False, 'message': 'Student ID is required'})
        
        data_manager = get_data_manager()
        student = data_manager.get_student(student_id)
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'})
        
        # Handle uploaded training images
        training_images = request.files.getlist('training_images')
        if not training_images or len(training_images) == 0:
            return jsonify({'success': False, 'message': 'At least one training image is required'})
        
        # Save uploaded images temporarily
        temp_paths = []
        training_folder = os.path.join(app.config['DATA_FOLDER'], 'training_images', student_id)
        os.makedirs(training_folder, exist_ok=True)
        
        for i, file in enumerate(training_images):
            if file and allowed_file(file.filename):
                filename = f"training_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(training_folder, filename)
                file.save(filepath)
                temp_paths.append(filepath)
        
        if not temp_paths:
            return jsonify({'success': False, 'message': 'No valid images were uploaded'})
        
        # Train with multiple images
        result = data_manager.train_face_with_multiple_images(student_id, temp_paths)
        
        # Update training statistics in response
        if result['success']:
            training_summary = data_manager.get_trained_faces_summary()
            result['training_summary'] = training_summary
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in batch face training: {e}")
        return jsonify({'success': False, 'message': f'Training failed: {str(e)}'})

@app.route('/get_training_progress/<student_id>')
@login_required
def get_training_progress(student_id):
    """Get training progress for a specific student"""
    try:
        data_manager = get_data_manager()
        progress = data_manager.get_face_training_progress(student_id)
        return jsonify(progress)
        
    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        return jsonify({'exists': False, 'error': str(e)})

@app.route('/retrain_face/<student_id>', methods=['POST'])
@login_required
def retrain_face(student_id):
    """Retrain face with additional images"""
    try:
        data_manager = get_data_manager()
        student = data_manager.get_student(student_id)
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'})
        
        # Get new training images
        new_images = request.files.getlist('additional_images')
        if not new_images:
            return jsonify({'success': False, 'message': 'No additional images provided'})
        
        # Save new images
        training_folder = os.path.join(app.config['DATA_FOLDER'], 'training_images', student_id)
        os.makedirs(training_folder, exist_ok=True)
        
        temp_paths = []
        for i, file in enumerate(new_images):
            if file and allowed_file(file.filename):
                filename = f"retrain_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(training_folder, filename)
                file.save(filepath)
                temp_paths.append(filepath)
        
        if not temp_paths:
            return jsonify({'success': False, 'message': 'No valid images were uploaded'})
        
        # Get existing training images
        existing_images = []
        if os.path.exists(training_folder):
            for filename in os.listdir(training_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    existing_images.append(os.path.join(training_folder, filename))
        
        # Combine existing and new images for retraining
        all_images = existing_images + temp_paths
        
        # Retrain with all images
        result = data_manager.train_face_with_multiple_images(student_id, all_images)
        
        if result['success']:
            training_summary = data_manager.get_trained_faces_summary()
            result['training_summary'] = training_summary
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in face retraining: {e}")
        return jsonify({'success': False, 'message': f'Retraining failed: {str(e)}'})

@app.route('/bulk_train_all', methods=['POST'])
@login_required
def bulk_train_all():
    """Bulk train faces for multiple students"""
    try:
        # This endpoint expects a JSON payload with student_id -> image_paths mapping
        training_data = request.get_json()
        if not training_data:
            return jsonify({'success': False, 'message': 'No training data provided'})
        
        data_manager = get_data_manager()
        result = data_manager.bulk_train_faces(training_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in bulk training: {e}")
        return jsonify({'success': False, 'message': f'Bulk training failed: {str(e)}'})

@app.route('/get_training_statistics')
@login_required
def get_training_statistics():
    """Get comprehensive training statistics"""
    try:
        data_manager = get_data_manager()
        training_summary = data_manager.get_trained_faces_summary()
        
        # Add additional statistics
        students = data_manager.get_all_students()
        total_students = len(students)
        trained_students = training_summary['statistics']['total_faces']
        untrained_students = total_students - trained_students
        
        enhanced_stats = {
            **training_summary,
            'additional_stats': {
                'total_students': total_students,
                'trained_students': trained_students,
                'untrained_students': untrained_students,
                'training_completion_rate': round((trained_students / total_students * 100) if total_students > 0 else 0, 1)
            }
        }
        
        return jsonify(enhanced_stats)
        
    except Exception as e:
        logger.error(f"Error getting training statistics: {e}")
        return jsonify({'error': str(e)})

@app.route('/clear_all_training', methods=['POST'])
@login_required
def clear_all_training():
    """Clear all face training data (admin only)"""
    try:
        data_manager = get_data_manager()
        
        # Clear all face embeddings
        data_manager.face_embeddings.clear()
        data_manager._save_face_embeddings()
        
        # Clear training metadata
        data_manager._save_training_metadata({})
        
        # Remove training images directories
        training_base_dir = os.path.join(app.config['DATA_FOLDER'], 'training_images')
        if os.path.exists(training_base_dir):
            import shutil
            shutil.rmtree(training_base_dir)
        
        return jsonify({'success': True, 'message': 'All training data cleared successfully'})
        
    except Exception as e:
        logger.error(f"Error clearing training data: {e}")
        return jsonify({'success': False, 'message': f'Failed to clear training data: {str(e)}'})

@app.route('/validate_training_images', methods=['POST'])
@login_required
def validate_training_images():
    """Validate uploaded training images before processing"""
    try:
        images = request.files.getlist('images')
        if not images:
            return jsonify({'success': False, 'message': 'No images provided'})
        
        face_recognizer = get_face_recognizer()
        validation_results = []
        
        for i, file in enumerate(images):
            if not file or not allowed_file(file.filename):
                validation_results.append({
                    'index': i,
                    'filename': file.filename if file else 'Unknown',
                    'valid': False,
                    'error': 'Invalid file type'
                })
                continue
            
            try:
                # Save temporarily for validation
                temp_filename = f"temp_validation_{i}.jpg"
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                file.save(temp_path)
                
                # Load and check for faces
                image = face_recognizer.load_image(temp_path)
                if image is None:
                    validation_results.append({
                        'index': i,
                        'filename': file.filename,
                        'valid': False,
                        'error': 'Could not load image'
                    })
                else:
                    embedding = face_recognizer.extract_face_embedding(image)
                    if embedding is not None:
                        validation_results.append({
                            'index': i,
                            'filename': file.filename,
                            'valid': True,
                            'message': 'Face detected successfully'
                        })
                    else:
                        validation_results.append({
                            'index': i,
                            'filename': file.filename,
                            'valid': False,
                            'error': 'No face detected in image'
                        })
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                validation_results.append({
                    'index': i,
                    'filename': file.filename,
                    'valid': False,
                    'error': f'Validation error: {str(e)}'
                })
        
        valid_count = sum(1 for result in validation_results if result['valid'])
        
        return jsonify({
            'success': True,
            'results': validation_results,
            'summary': {
                'total': len(images),
                'valid': valid_count,
                'invalid': len(images) - valid_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error validating training images: {e}")
        return jsonify({'success': False, 'message': f'Validation failed: {str(e)}'})

@app.errorhandler(413)
def file_too_large(error):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)
