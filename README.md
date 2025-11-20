# Face Recognition Attendance System

A comprehensive web-based face recognition attendance system built with Flask, OpenCV, and cutting-edge deep learning technologies. This system provides automatic attendance marking through facial recognition with multiple advanced detection and recognition methods.

## 🌟 Features

### Core Functionality
- **Real-time Face Recognition**: Live camera-based attendance marking
- **Bulk Photo Processing**: Automatic attendance from group photos
- **Student Management**: Complete student registration and management system
- **Face Training System**: Multi-image training for improved accuracy
- **Attendance Tracking**: Comprehensive attendance records with analytics

### Advanced Face Recognition Technologies
- **MTCNN Detection**: Multi-task Convolutional Neural Network for precise face detection
- **Custom CNN + ArcFace**: State-of-the-art face recognition with ArcFace loss function
- **Ensemble Matching**: Combines cosine similarity, SVM, and Euclidean distance
- **Intelligent Fallbacks**: Graceful degradation to OpenCV and basic methods
- **Technology Selection**: Choose specific detection and recognition methods

### Additional Features
- **Location-Aware Attendance**: Automatic GPS extraction from geo-tagged photos
- **Multiple Recognition Engines**: Fallback systems for maximum reliability
- **Secure Data Management**: Encrypted data storage and validation
- **Responsive Web Interface**: Mobile-friendly design with technology selection
- **Export Capabilities**: CSV export for attendance and student data

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher (tested with Python 3.13)
- Webcam (for real-time recognition)
- Modern web browser

### Installation Options

#### Option 1: Automated Installation (Recommended)

**Windows:**
```cmd
install.bat
```

**Linux/Mac:**
```bash
python install.py
```

#### Option 2: Traditional Deployment

**Windows:**
```cmd
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

#### Option 3: Manual Installation

1. **Install core dependencies**
   ```bash
   pip install -r requirements_core.txt
   ```

2. **Install advanced features (optional)**
   ```bash
   pip install -r requirements_advanced.txt
   ```

3. **Create models directory**
   ```bash
   mkdir models
   ```
4. **Configure environment**
   - Edit `.env` file with your production settings
   - Change default admin password
   - Set DEBUG=False for production

5. **Start the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:5000`
   - Login with admin credentials

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended for advanced features)
- **Storage**: 2GB free space
- **CPU**: Multi-core processor recommended
- **Camera**: USB webcam or built-in camera

### Core Dependencies
```
Flask>=2.3.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
face-recognition>=1.3.0
```

### Advanced Dependencies (Optional)
```
torch>=2.0.0
torchvision>=0.15.0
facenet-pytorch>=2.5.3
matplotlib>=3.7.0
tqdm>=4.65.0
```

## 🧠 Technology Stack

### Detection Methods
1. **MTCNN (Multi-task CNN)**: Advanced face detection with facial landmarks
2. **Custom CNN**: Deep learning-based detection
3. **OpenCV Haar Cascades**: Traditional computer vision fallback

### Recognition Methods
1. **Custom CNN + ArcFace Loss**: State-of-the-art face recognition
2. **Face Recognition Library**: dlib-based recognition
3. **Basic Feature Extraction**: OpenCV-based fallback

### Matching Algorithms
1. **Cosine Similarity**: Vector similarity matching
2. **SVM Classifier**: Machine learning classification
3. **Euclidean Distance**: Traditional distance-based matching
4. **Ensemble Method**: Combines all methods with cross-validation

## 🔧 Configuration

### Environment Variables (.env)
```env
SECRET_KEY=your-very-secure-secret-key-change-this-in-production
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change-this-secure-password
DEBUG=False
UPLOAD_FOLDER=uploads
DATA_FOLDER=data
MAX_CONTENT_LENGTH=16777216
FLASK_ENV=production
```

### Security Settings
- Change default admin credentials
- Use strong SECRET_KEY in production
- Set DEBUG=False for production deployment
- Configure HTTPS for production use

## 📖 Usage Guide

### 1. Student Registration
- Navigate to "Student Management"
- Click "Add New Student"
- Fill in student details
- Upload student photo

### 2. Face Training

**Basic Training:**
- Go to "Face Training" section
- Upload multiple photos of each student
- Train the system for better accuracy
- Monitor training progress

**Advanced Training:**
- Access "Advanced Face Training" for cutting-edge features
- Select detection method: MTCNN, Custom CNN, or OpenCV
- Choose recognition method: Custom CNN+ArcFace, face_recognition, or basic
- Configure matching algorithm: Ensemble, cosine similarity, SVM, or Euclidean
- Monitor training with detailed progress and metrics

### 3. Attendance Marking

**Real-time Recognition:**
- Go to "Face Recognition"
- Start camera
- Enable real-time recognition
- Students will be automatically recognized

**Bulk Photo Processing:**
- Upload group photo
- Enable location extraction (optional)
- System processes all faces
- Attendance marked automatically

### 4. Attendance Management
- View daily attendance reports
- Export data as CSV
- Filter by date, student, or department
- View location-based attendance (if available)

## 🗂️ Project Structure

```
Face-Recognition/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements_core.txt           # Essential dependencies
├── requirements_advanced.txt       # Advanced ML dependencies
├── install.py                      # Python installation script
├── install.bat                     # Windows installation script
├── deploy.bat/sh                   # Deployment scripts
├── .env.example                    # Environment template
├── data/                          # Data storage
│   ├── students.csv               # Student records
│   ├── attendance.csv             # Attendance records
│   └── face_embeddings.pkl        # Face recognition data
├── models/                        # AI models and encodings
├── static/                        # Static web assets
│   ├── css/                       # Stylesheets
│   ├── js/                        # JavaScript files
│   └── images/                    # Image assets
├── templates/                     # HTML templates
│   ├── index.html                 # Main interface
│   └── asset_training.html        # Advanced training interface
├── utils/                         # Utility modules
│   ├── data_manager.py            # Data management
│   ├── enhanced_face_recognition.py # Basic face recognition
│   ├── advanced_face_detection.py   # MTCNN + Custom CNN detection
│   ├── advanced_face_recognition.py # Custom CNN + ArcFace recognition
│   ├── advanced_face_matching.py    # Ensemble matching algorithms
│   ├── integrated_face_system.py    # Complete pipeline integration
│   ├── advanced_training_routes.py  # Flask routes for advanced features
│   ├── geo_location.py            # Location services
│   └── security.py                # Security utilities
└── uploads/                       # Temporary upload storage
```

## 🔐 Security Features

- **Data Encryption**: Secure CSV handling with validation
- **Input Validation**: Comprehensive data validation
- **Session Management**: Secure user sessions
- **File Upload Security**: Safe file handling with type validation
- **SQL Injection Protection**: Parameterized queries (when applicable)

## 🌍 Location Features

### GPS Data Extraction
- Automatic extraction from geo-tagged photos
- Reverse geocoding for readable addresses
- Multiple geocoding service support
- Privacy-focused implementation

### Location Data Storage
- Coordinates stored with attendance records
- Human-readable address resolution
- City, state, country information
- Optional location extraction

## 📊 Analytics & Reporting

- Daily attendance summaries
- Student-wise attendance tracking
- Department and year-wise reports
- Location-based attendance patterns
- CSV export for external analysis

## 🛠️ Troubleshooting

### Installation Issues

**Python 3.13 Compatibility:**
- Use `requirements_core.txt` for essential packages
- Install advanced packages individually if needed
- The system provides fallbacks for missing dependencies

**PyTorch Installation:**
```bash
# For CPU-only (recommended for most users)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA GPU support (if you have compatible GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Face Recognition Library Issues:**
```bash
# On Windows, install Microsoft C++ Build Tools first
# Alternative approach:
conda install -c conda-forge dlib
pip install face-recognition
```

**Missing Dependencies:**
```bash
# If advanced features fail, the system will fall back to basic OpenCV
# Check the console for specific error messages
```

### Common Runtime Issues

**Camera not working:**
- Check camera permissions
- Ensure camera is not in use by other applications
- Try refreshing the page

**Student photos not displaying:**
- Photos will show default avatar if no photo is available
- Check that student photos are properly uploaded during registration
- System automatically falls back to default avatar for missing photos
- Photos are stored in `data/photos/` directory for permanent storage

**Face recognition accuracy issues:**
- Use the Face Training feature with multiple photos per student
- Ensure good lighting conditions
- Upload clear, front-facing photos
- Retrain the system after adding new students

**Poor recognition accuracy:**
- Use face training feature with multiple photos
- Ensure good lighting conditions
- Use clear, high-quality images

**Location not detected:**
- Ensure photos are geo-tagged
- Check internet connection for geocoding
- Verify GPS was enabled when photo was taken

### Debug Mode
For development, set `DEBUG=True` in `.env` file to see detailed error messages.

## 📱 Browser Support

- **Chrome/Chromium** 88+ (Recommended)
- **Firefox** 85+
- **Safari** 14+
- **Edge** 88+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🔄 Updates

The system supports automatic updates for:
- Student data management
- Face recognition improvements
- Security enhancements
- New feature additions

---

**Note**: This system is designed for educational and small-scale commercial use. For large-scale deployments, consider additional security measures and performance optimizations.
