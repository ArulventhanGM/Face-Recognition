# Face Recognition Attendance System

A comprehensive web-based face recognition attendance system built with Flask, OpenCV, and modern web technologies. This system provides automatic attendance marking through facial recognition, student management, and location-aware attendance tracking.

## 🌟 Features

### Core Functionality
- **Real-time Face Recognition**: Live camera-based attendance marking
- **Bulk Photo Processing**: Automatic attendance from group photos
- **Student Management**: Complete student registration and management system
- **Face Training System**: Multi-image training for improved accuracy
- **Attendance Tracking**: Comprehensive attendance records with analytics

### Advanced Features
- **Location-Aware Attendance**: Automatic GPS extraction from geo-tagged photos
- **Multiple Recognition Engines**: Fallback systems for maximum reliability
- **Secure Data Management**: Encrypted data storage and validation
- **Responsive Web Interface**: Mobile-friendly design
- **Export Capabilities**: CSV export for attendance and student data

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time recognition)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArulventhanGM/Face-Recognition.git
   cd Face-Recognition
   ```

2. **Run deployment script**
   
   **Windows:**
   ```cmd
   deploy.bat
   ```
   
   **Linux/Mac:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Configure environment**
   - Edit `.env` file with your production settings
   - Change default admin password
   - Set DEBUG=False for production

4. **Start the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser to `http://localhost:5000`
   - Login with admin credentials

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **CPU**: Multi-core processor recommended
- **Camera**: USB webcam or built-in camera

### Dependencies
```
Flask>=2.3.3
opencv-python>=4.8.1
numpy>=1.24.3
pandas>=2.0.3
Pillow>=10.0.0
exifread>=3.0.0
geopy>=2.4.0
requests>=2.31.0
```

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

### 2. Face Training (Recommended)
- Go to "Face Training" section
- Upload multiple photos of each student
- Train the system for better accuracy
- Monitor training progress

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
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── deploy.bat/sh          # Deployment scripts
├── .env.example          # Environment template
├── data/                 # Data storage
│   ├── students.csv      # Student records
│   ├── attendance.csv    # Attendance records
│   └── face_embeddings.pkl # Face recognition data
├── static/               # Static web assets
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── images/          # Image assets
├── templates/           # HTML templates
├── utils/               # Utility modules
│   ├── data_manager.py  # Data management
│   ├── enhanced_face_recognition.py # Face recognition
│   ├── geo_location.py  # Location services
│   └── security.py      # Security utilities
└── uploads/             # Temporary upload storage
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

### Common Issues

**Camera not working:**
- Check camera permissions
- Ensure camera is not in use by other applications
- Try refreshing the page

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
