# Deployment Checklist

## Pre-Deployment Setup

### 1. Environment Configuration
- [ ] Copy `.env.example` to `.env`
- [ ] Set `DEBUG=False` in `.env`
- [ ] Change default `ADMIN_PASSWORD` to a secure password
- [ ] Generate a strong `SECRET_KEY` (min 32 characters)
- [ ] Set `FLASK_ENV=production`

### 2. Security Configuration
- [ ] Change admin username from default
- [ ] Use HTTPS in production
- [ ] Set strong session cookies
- [ ] Review file upload restrictions
- [ ] Validate input sanitization

### 3. System Requirements
- [ ] Python 3.8+ installed
- [ ] Camera/webcam available (for real-time recognition)
- [ ] Minimum 4GB RAM
- [ ] 2GB free disk space
- [ ] Modern web browser

### 4. Dependencies
- [ ] Run `pip install -r requirements.txt`
- [ ] Verify OpenCV installation
- [ ] Test camera access
- [ ] Check internet connectivity (for geocoding)

## Deployment Options

### Option 1: Direct Python Deployment
```bash
# Run deployment script
./deploy.sh  # Linux/Mac
deploy.bat   # Windows

# Start application
python app.py
```

### Option 2: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs face-recognition
```

### Option 3: Production Server (Gunicorn)
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

## Post-Deployment Verification

### 1. Application Health
- [ ] Access application at configured URL
- [ ] Check health endpoint: `/health`
- [ ] Verify admin login works
- [ ] Test student registration
- [ ] Test face recognition functionality

### 2. File System
- [ ] Verify `uploads/` directory exists and is writable
- [ ] Verify `data/` directory exists and is writable
- [ ] Check CSV files are created correctly
- [ ] Test file upload functionality

### 3. Security
- [ ] Admin login requires correct credentials
- [ ] Session management works correctly
- [ ] File uploads are restricted to allowed types
- [ ] No debug information exposed

### 4. Features
- [ ] Student registration works
- [ ] Photo uploads process correctly
- [ ] Face recognition detects faces
- [ ] Attendance marking functions
- [ ] CSV exports work
- [ ] Location extraction works (if enabled)

## Production Optimizations

### 1. Performance
- [ ] Enable production logging
- [ ] Configure log rotation
- [ ] Set appropriate worker count
- [ ] Optimize image processing settings

### 2. Monitoring
- [ ] Set up health checks
- [ ] Configure error logging
- [ ] Monitor disk usage
- [ ] Monitor memory usage

### 3. Backup
- [ ] Backup `data/` directory regularly
- [ ] Backup face embeddings
- [ ] Backup configuration files
- [ ] Set up automated backups

## Troubleshooting

### Common Issues

1. **Camera not working**
   - Check camera permissions
   - Verify camera not in use by other applications
   - Test camera access outside the application

2. **Poor face recognition accuracy**
   - Use face training feature with multiple photos
   - Ensure good lighting conditions
   - Use clear, high-quality images

3. **File upload errors**
   - Check disk space
   - Verify upload directory permissions
   - Confirm file size within limits

4. **Location features not working**
   - Verify internet connection
   - Check if images contain GPS metadata
   - Review geocoding service availability

### Log Files
- Application logs: `face_recognition.log`
- System logs: Check system log files
- Docker logs: `docker-compose logs`

## Security Checklist

### 1. Authentication
- [ ] Strong admin password set
- [ ] Session timeout configured
- [ ] Login attempts limited

### 2. File Security
- [ ] File upload restrictions in place
- [ ] File type validation active
- [ ] Upload directory secured

### 3. Data Protection
- [ ] CSV files protected
- [ ] Face embeddings secured
- [ ] Temporary files cleaned up

### 4. Network Security
- [ ] HTTPS enabled (production)
- [ ] Firewall configured
- [ ] Unnecessary ports closed

## Maintenance

### Regular Tasks
- [ ] Monitor disk usage
- [ ] Review log files
- [ ] Update dependencies (security patches)
- [ ] Backup data files
- [ ] Clean temporary files

### Updates
- [ ] Test updates in development first
- [ ] Backup before updates
- [ ] Review changelog for breaking changes
- [ ] Verify functionality after updates

## Support

### Documentation
- README.md for general usage
- Code comments for technical details
- API documentation for integration

### Monitoring
- Health endpoint: `/health`
- Application logs
- System resource monitoring

### Backup and Recovery
- Regular data backups
- Configuration backups
- Recovery procedures documented

---

**Note**: This checklist should be customized based on your specific deployment environment and requirements.
