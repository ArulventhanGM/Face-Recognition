# Face Recognition Academic System

A comprehensive face recognition web application designed for academic institutions, featuring real-time attendance tracking, photo-based recognition, and secure admin management.

## Features

### 🔐 Admin Access & Security
- Secure admin-only login system
- Protected CSV data operations with validation
- Input sanitization and CSV injection protection

### 🎓 Academic Functionality
- Student registration with face data
- Real-time webcam face recognition
- Batch photo processing for attendance
- Automated attendance marking
- Downloadable CSV reports

### 🧠 Face Recognition Technology
- ArcFace algorithm for high-accuracy recognition
- Robust facial embeddings
- Live and uploaded photo processing

### 🎨 Modern UI Design
- Responsive design with smooth transitions
- Cool Blue (#3498db) color scheme
- Intuitive user interface
- Real-time feedback

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ArulventhanGM/Face-Recognition.git
cd Face-Recognition
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
copy .env.example .env
# Edit .env with your configuration
```

5. Run the application:
```bash
python app.py
```

## Usage

1. Access the application at `http://localhost:5000`
2. Login with admin credentials
3. Register students with face data
4. Use real-time recognition or upload group photos
5. Download attendance reports as CSV

## File Structure

```
Face-Recognition/
├── app.py                  # Main Flask application
├── config.py              # Configuration settings
├── models/                 # Face recognition models
├── static/                 # CSS, JS, images
├── templates/             # HTML templates
├── data/                  # CSV storage
├── uploads/               # Uploaded images
└── utils/                 # Utility functions
```

## Security Features

- Admin authentication
- CSV injection protection
- Input validation
- Secure file handling
- Session management

## License

MIT License
