from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import os
import cv2
import numpy as np
import base64
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter

from config import Config
from utils.data_manager import get_data_manager
try:
    from utils.face_recognition_utils import get_face_recognizer
except ImportError:
    # Fallback to mock for testing
    from utils.face_recognition_mock import get_face_recognizer
from utils.security import sanitize_filename, validate_student_id, validate_email

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
            
            # Validate
            if not all(updated_data.values()):
                flash('All fields are required', 'error')
                return render_template('add_student.html', student=student)
            
            if not validate_email(updated_data['email']):
                flash('Invalid email format', 'error')
                return render_template('add_student.html', student=student)
            
            # Handle face image update
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
                                flash('Face data updated successfully!', 'success')
                            else:
                                flash('Could not extract face from image', 'warning')
                        
                        # Clean up uploaded file
                        try:
                            os.remove(face_image_path)
                        except:
                            pass
            
            # Update student data
            success = data_manager.update_student(student_id, updated_data)
            
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
            recognized_faces = data_manager.recognize_face_from_image(filepath)
            
            return jsonify({
                'success': True,
                'data': recognized_faces,
                'message': f'Recognized {len(recognized_faces)} face(s)'
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
    """Real-time face recognition from camera"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data'})
        
        # Decode base64 image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image data'})
        
        # Extract face embeddings
        face_recognizer = get_face_recognizer()
        face_embeddings = face_recognizer.extract_multiple_face_embeddings(image)
        
        recognized_faces = []
        data_manager = get_data_manager()
        
        for embedding in face_embeddings:
            result = data_manager.recognize_face_from_embedding(embedding)
            if result:
                recognized_faces.append(result)
        
        return jsonify({
            'success': True,
            'data': recognized_faces
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
    """Mark attendance from group photo"""
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
            # Process group photo for attendance
            data_manager = get_data_manager()
            results = data_manager.bulk_mark_attendance_from_image(filepath)
            
            return jsonify({
                'success': True,
                'data': results,
                'message': 'Bulk attendance processing completed'
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        logger.error(f"Error processing group photo: {e}")
        return jsonify({'success': False, 'message': 'Failed to process group photo'})

@app.route('/student_photo/<student_id>')
@login_required
def student_photo(student_id):
    """Serve student photo"""
    try:
        data_manager = get_data_manager()
        student = data_manager.get_student(student_id)
        
        if not student:
            # Return default avatar
            return redirect(url_for('static', filename='images/default-avatar.svg'))
        
        # Look for student photo in uploads folder
        upload_folder = app.config['UPLOAD_FOLDER']
        
        # Check for common image extensions
        for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            photo_path = os.path.join(upload_folder, f"{student_id}.{ext}")
            if os.path.exists(photo_path):
                return send_file(photo_path)
        
        # Look for timestamped files containing student_id
        for filename in os.listdir(upload_folder):
            if student_id.lower() in filename.lower() and any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
                photo_path = os.path.join(upload_folder, filename)
                return send_file(photo_path)
        
        # Return default avatar if no photo found
        return redirect(url_for('static', filename='images/default-avatar.svg'))
        
    except Exception as e:
        logger.error(f"Error serving student photo for {student_id}: {e}")
        return redirect(url_for('static', filename='images/default-avatar.svg'))

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

@app.errorhandler(413)
def file_too_large(error):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)
