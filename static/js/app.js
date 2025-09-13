// Face Recognition Academic System - JavaScript

class FaceRecognitionApp {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.context = null;
        this.stream = null;
        this.isRecognitionActive = false;
        this.recognitionInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeComponents();
    }

    setupEventListeners() {
        // File upload handling
        const fileInputs = document.querySelectorAll('input[type="file"]');
        fileInputs.forEach(input => {
            input.addEventListener('change', this.handleFileUpload.bind(this));
        });

        // Form submissions
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        });

        // Camera controls
        const startCameraBtn = document.getElementById('startCamera');
        const stopCameraBtn = document.getElementById('stopCamera');
        const captureBtn = document.getElementById('capturePhoto');
        const toggleRecognitionBtn = document.getElementById('toggleRecognition');

        if (startCameraBtn) {
            startCameraBtn.addEventListener('click', this.startCamera.bind(this));
        }
        
        if (stopCameraBtn) {
            stopCameraBtn.addEventListener('click', this.stopCamera.bind(this));
        }

        if (captureBtn) {
            captureBtn.addEventListener('click', this.capturePhoto.bind(this));
        }

        if (toggleRecognitionBtn) {
            toggleRecognitionBtn.addEventListener('click', this.toggleRecognition.bind(this));
        }

        // Delete confirmations
        const deleteButtons = document.querySelectorAll('.btn-danger[data-confirm]');
        deleteButtons.forEach(btn => {
            btn.addEventListener('click', this.confirmDelete.bind(this));
        });
    }

    initializeComponents() {
        // Initialize video element
        this.video = document.getElementById('cameraPreview');
        if (this.video) {
            this.canvas = document.createElement('canvas');
            this.context = this.canvas.getContext('2d');
            
            // Set additional video attributes for better compatibility
            this.video.setAttribute('playsinline', 'true');  // Important for iOS Safari
            this.video.setAttribute('webkit-playsinline', 'true');  // For older iOS
            this.video.setAttribute('muted', 'true');
            this.video.muted = true;  // Ensure muted for autoplay
        }

        // Auto-hide alerts
        this.autoHideAlerts();
        
        // Initialize tooltips if needed
        this.initializeTooltips();
        
        // Check for secure context - required for camera access in modern browsers
        if (this.video && !window.isSecureContext) {
            console.warn('Not running in a secure context. Camera may not work.');
            this.showAlert('Warning: This page must be accessed via HTTPS for camera functionality.', 'warning', 10000);
        }
    }

    async startCamera() {
        // Check if browser supports getUserMedia
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showAlert('Your browser does not support camera access. Please use Chrome, Firefox, or Edge.', 'error');
            return;
        }
        
        try {
            this.showLoading('startCamera');
            console.log('Starting camera initialization...');
            
            // Reset any previous stream
            if (this.stream) {
                this.stopCamera();
            }
            
            // Check for camera permissions first
            try {
                // Try permissions check if available
                if (navigator.permissions && navigator.permissions.query) {
                    const permissionStatus = await navigator.permissions.query({ name: 'camera' });
                    console.log('Camera permission status:', permissionStatus.state);
                    
                    if (permissionStatus.state === 'denied') {
                        throw new Error('Camera permission denied');
                    }
                }
            } catch (permError) {
                console.log('Permission check failed, continuing anyway:', permError);
                // Continue anyway, getUserMedia will handle permissions
            }
            
            // Try to enumerate devices to check for cameras
            let videoDevices = [];
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                videoDevices = devices.filter(device => device.kind === 'videoinput');
                console.log('Available video devices:', videoDevices.length);
                
                if (videoDevices.length === 0) {
                    console.warn('No video devices found in enumeration');
                    // We'll still try getUserMedia as enumeration might be limited without permission
                }
            } catch (enumError) {
                console.warn('Could not enumerate devices:', enumError);
                // Continue anyway, as some browsers may not allow enumeration without permission
            }
            
            // Progressive fallback for camera access
            const constraints = [
                // First try: HD with specific parameters
                {
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: 'user'
                    }
                },
                // Second try: Lower resolution
                {
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                },
                // Third try: Very basic constraints
                { video: true },
                // Fourth try: Environment facing camera (rear camera on phones)
                {
                    video: {
                        facingMode: 'environment'
                    }
                }
            ];
            
            // Try each constraint in sequence until one works
            let stream = null;
            let lastError = null;
            
            for (let i = 0; i < constraints.length; i++) {
                try {
                    console.log(`Trying camera constraints option ${i+1}:`, constraints[i]);
                    stream = await navigator.mediaDevices.getUserMedia(constraints[i]);
                    console.log(`Successfully got stream with option ${i+1}`);
                    break; // Success!
                } catch (error) {
                    console.warn(`Option ${i+1} failed:`, error);
                    lastError = error;
                    // Continue to next constraint option
                }
            }
            
            if (!stream) {
                throw lastError || new Error('Could not access any camera');
            }
            
            this.stream = stream;
            
            // Connect stream to video element
            if (this.video) {
                console.log('Connecting stream to video element');
                this.video.srcObject = this.stream;
                
                // Create a promise for video loaded
                const playPromise = new Promise((resolve, reject) => {
                    this.video.onloadedmetadata = () => {
                        this.video.play()
                            .then(() => resolve())
                            .catch(err => reject(err));
                    };
                    
                    this.video.onerror = (err) => {
                        reject(err);
                    };
                    
                    // Set a timeout in case onloadedmetadata never fires
                    setTimeout(() => {
                        reject(new Error('Video loading timeout'));
                    }, 10000);
                });
                
                await playPromise;
                
                console.log('Video playback started');
                this.updateCameraControls(true);
                this.hideLoading('startCamera');
                this.showAlert('Camera started successfully', 'success');
                
                // Log camera details for debugging
                const tracks = this.stream.getVideoTracks();
                if (tracks.length > 0) {
                    const settings = tracks[0].getSettings();
                    console.log('Active camera settings:', settings);
                }
            }
        } catch (error) {
            console.error('Error starting camera:', error);
            this.hideLoading('startCamera');
            
            let errorMessage = 'Failed to start camera.';
            if (error.name === 'NotFoundError') {
                errorMessage = 'No camera found. Please make sure a camera is connected to your device.';
                this.showCameraTroubleshooting();
            } else if (error.name === 'NotAllowedError') {
                errorMessage = 'Camera access denied. Please allow camera access in your browser settings.';
                this.showPermissionTroubleshooting();
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Camera is already in use by another application. Please close any other apps using your camera.';
            } else if (error.name === 'OverconstrainedError') {
                errorMessage = 'Your camera does not support the required settings.';
            } else if (error.name === 'AbortError') {
                errorMessage = 'Camera access was aborted. Please try again.';
            } else if (error.name === 'SecurityError') {
                errorMessage = 'Camera access blocked due to security restrictions. Please use HTTPS.';
            } else if (error.message && error.message.includes('permission')) {
                errorMessage = 'Camera permission denied. Please allow camera access in your browser settings.';
                this.showPermissionTroubleshooting();
            }
            
            this.showAlert(errorMessage, 'error', 10000);
        }
    }
    
    showCameraTroubleshooting() {
        const resultsPanel = document.getElementById('realTimeResults');
        if (resultsPanel) {
            resultsPanel.innerHTML = `
                <div class="troubleshooting-guide">
                    <h4>Camera Troubleshooting</h4>
                    <ul>
                        <li>Make sure your camera is properly connected</li>
                        <li>Verify no other application is using your camera</li>
                        <li>Try refreshing the page</li>
                        <li>Try a different browser (Chrome or Firefox recommended)</li>
                        <li>On Windows, check Device Manager to ensure camera is working</li>
                        <li>On Mac, check System Preferences > Security & Privacy</li>
                    </ul>
                </div>
            `;
        }
    }
    
    showPermissionTroubleshooting() {
        const resultsPanel = document.getElementById('realTimeResults');
        if (resultsPanel) {
            resultsPanel.innerHTML = `
                <div class="troubleshooting-guide">
                    <h4>Camera Permission Guide</h4>
                    <p>Your browser is blocking camera access. To fix:</p>
                    <ul>
                        <li>Look for the camera icon in your address bar</li>
                        <li>Click it and select "Allow" for camera access</li>
                        <li>Refresh this page after changing permissions</li>
                        <li>In Chrome: Settings > Privacy and Security > Site Settings > Camera</li>
                        <li>In Firefox: Preferences > Privacy & Security > Permissions > Camera</li>
                        <li>In Edge: Settings > Site Permissions > Camera</li>
                    </ul>
                </div>
            `;
        }
    }

    stopCamera() {
        try {
            if (this.stream) {
                console.log('Stopping camera tracks');
                this.stream.getTracks().forEach(track => {
                    try {
                        track.stop();
                        console.log(`Track ${track.id} stopped`);
                    } catch (e) {
                        console.warn(`Error stopping track ${track.id}:`, e);
                    }
                });
                this.stream = null;
            }
            
            if (this.video) {
                console.log('Clearing video source');
                this.video.pause();
                this.video.srcObject = null;
            }

            this.stopRecognition();
            this.updateCameraControls(false);
            this.showAlert('Camera stopped', 'info');
            
            // Clear any troubleshooting guides
            const resultsPanel = document.getElementById('realTimeResults');
            if (resultsPanel) {
                resultsPanel.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 20px;">Start recognition to see results</p>';
            }
        } catch (error) {
            console.error('Error stopping camera:', error);
        }
    }

    capturePhoto() {
        if (!this.video || !this.stream) {
            this.showAlert('Camera not active', 'error');
            return;
        }

        try {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.context.drawImage(this.video, 0, 0);
            
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server for processing
            this.processPhoto(imageData);
            
        } catch (error) {
            console.error('Error capturing photo:', error);
            this.showAlert('Failed to capture photo', 'error');
        }
    }

    async processPhoto(imageData) {
        try {
            this.showLoading('capturePhoto');
            
            const response = await fetch('/process_photo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayRecognitionResults(result);

                // Show appropriate success message based on results
                if (result.faces_detected === 0) {
                    this.showAlert('No faces detected in the uploaded image', 'warning');
                } else if (result.faces_recognized === 0) {
                    this.showAlert(`Detected ${result.faces_detected} face(s) but none were recognized`, 'warning');
                } else {
                    this.showAlert(`Successfully processed ${result.faces_detected} face(s), recognized ${result.faces_recognized}`, 'success');
                }
            } else {
                this.showAlert(result.message || 'Failed to process photo', 'error');
            }
            
        } catch (error) {
            console.error('Error processing photo:', error);
            this.showAlert('Failed to process photo', 'error');
        } finally {
            this.hideLoading('capturePhoto');
        }
    }

    toggleRecognition() {
        if (this.isRecognitionActive) {
            this.stopRecognition();
        } else {
            this.startRecognition();
        }
    }

    startRecognition() {
        if (!this.video || !this.stream) {
            this.showAlert('Camera not active', 'error');
            return;
        }

        this.isRecognitionActive = true;
        this.updateRecognitionButton();
        
        // Start recognition loop
        this.recognitionInterval = setInterval(() => {
            this.performRecognition();
        }, 1000); // Recognize every second

        this.showAlert('Real-time recognition started', 'success');
    }

    stopRecognition() {
        this.isRecognitionActive = false;
        this.updateRecognitionButton();
        
        if (this.recognitionInterval) {
            clearInterval(this.recognitionInterval);
            this.recognitionInterval = null;
        }

        this.clearRecognitionResults();
        this.showAlert('Real-time recognition stopped', 'info');
    }

    async performRecognition() {
        if (!this.video || !this.isRecognitionActive) return;

        try {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.context.drawImage(this.video, 0, 0);
            
            const imageData = this.canvas.toDataURL('image/jpeg', 0.5);
            
            const response = await fetch('/recognize_realtime', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const result = await response.json();
            
            if (result.success && result.data.length > 0) {
                this.displayRealTimeResults(result.data);
            } else {
                this.clearRecognitionResults();
            }
            
        } catch (error) {
            console.error('Error in real-time recognition:', error);
        }
    }

    displayRecognitionResults(response) {
        console.log('displayRecognitionResults called with:', response); // Debug log
        const container = document.getElementById('recognitionResults');
        if (!container) {
            console.error('recognitionResults container not found!');
            return;
        }

        container.innerHTML = '';

        // Handle case where no faces were detected
        if (response.faces_detected === 0) {
            container.innerHTML = `
                <div class="no-faces-container">
                    <div class="alert alert-info">
                        <i class="fas fa-search"></i>
                        <strong>No faces detected in the image</strong><br>
                        Please upload an image that contains one or more faces.
                    </div>
                    <div class="text-center mt-3">
                        <button class="btn btn-primary" onclick="app.resetUploadForm()">
                            <i class="fas fa-upload"></i> Upload Another Image
                        </button>
                    </div>
                </div>
            `;
            return;
        }

        // Handle case where faces were detected but none recognized
        if (response.faces_recognized === 0) {
            container.innerHTML = `
                <div class="no-matches-container">
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle"></i>
                        <strong>No matches found</strong><br>
                        Detected ${response.faces_detected} face(s) but none match registered students.
                    </div>
                    <div class="text-center mt-3">
                        <button class="btn btn-primary" onclick="app.showAddNewUserDialog()">
                            <i class="fas fa-user-plus"></i> Add New Student
                        </button>
                        <button class="btn btn-secondary ml-2" onclick="app.resetUploadForm()">
                            <i class="fas fa-upload"></i> Upload Another Image
                        </button>
                    </div>
                </div>
            `;
            return;
        }

        // Support both response.data and response.faces_recognized as sources for recognized faces
        let results = [];
        if (response.data && Array.isArray(response.data)) {
            results = response.data;
        } else if (response.faces_recognized && Array.isArray(response.faces_recognized)) {
            results = response.faces_recognized;
        }

        // Create results header with comprehensive information
        const header = document.createElement('div');
        header.className = 'results-header';
        header.innerHTML = `
            <h4><i class="fas fa-search"></i> Recognition Results</h4>
            <div class="results-summary">
                <div class="summary-stats">
                    <span class="stat-item">
                        <i class="fas fa-eye"></i> ${response.faces_detected} face(s) detected
                    </span>
                    <span class="stat-item">
                        <i class="fas fa-check-circle"></i> ${response.faces_recognized} recognized
                    </span>
                    ${response.faces_unrecognized > 0 ? `
                        <span class="stat-item">
                            <i class="fas fa-question-circle"></i> ${response.faces_unrecognized} unrecognized
                        </span>
                    ` : ''}
                </div>
            </div>
        `;
        container.appendChild(header);

        // Sort results by confidence (highest first)
        results.sort((a, b) => b.confidence - a.confidence);

        results.forEach((result, index) => {
            const resultItem = this.createEnhancedResultItem(result, index === 0);
            container.appendChild(resultItem);
        });

        // Add unrecognized faces information if any
        if (response.faces_unrecognized > 0) {
            const unrecognizedContainer = document.createElement('div');
            unrecognizedContainer.className = 'unrecognized-faces-container mt-3';
            unrecognizedContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i>
                    <strong>Unrecognized Faces</strong><br>
                    ${response.faces_unrecognized} face(s) were detected but not found in the student database.
                </div>
                <div class="text-center">
                    <button class="btn btn-outline-primary" onclick="app.showAddNewUserDialog()">
                        <i class="fas fa-user-plus"></i> Register New Student
                    </button>
                </div>
            `;
            container.appendChild(unrecognizedContainer);
        }

        // Add "Add New Student" option if confidence is low
        const bestMatch = results[0];
        if (bestMatch && bestMatch.confidence < 0.8) {
            const addNewContainer = document.createElement('div');
            addNewContainer.className = 'add-new-container mt-3';
            addNewContainer.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Low confidence match</strong><br>
                    If this person is not ${bestMatch.name}, you can register them as a new student.
                </div>
                <div class="text-center">
                    <button class="btn btn-outline-primary" onclick="app.showAddNewUserDialog()">
                        <i class="fas fa-user-plus"></i> Register as New Student
                    </button>
                </div>
            `;
            container.appendChild(addNewContainer);
        }

        // Add re-upload option
        const reuploadContainer = document.createElement('div');
        reuploadContainer.className = 'reupload-container mt-3 text-center';
        reuploadContainer.innerHTML = `
            <button class="btn btn-outline-secondary" onclick="app.resetUploadForm()">
                <i class="fas fa-upload"></i> Upload Another Image
            </button>
        `;
        container.appendChild(reuploadContainer);
    }

    displayRealTimeResults(results) {
        const container = document.getElementById('realTimeResults');
        if (!container) return;

        // If no results, show appropriate message
        if (!results || results.length === 0) {
            container.innerHTML = `
                <div class="no-results">
                    <p class="text-center text-muted">
                        <i class="fas fa-search"></i> Scanning for faces...
                    </p>
                </div>
            `;
            return;
        }

        // Sort results by confidence (highest first) to show best match
        results.sort((a, b) => b.confidence - a.confidence);

        container.innerHTML = '';
        
        // Add header showing scan status
        const header = document.createElement('div');
        header.className = 'realtime-header';
        header.innerHTML = `
            <h5><i class="fas fa-eye"></i> Live Recognition</h5>
            <small class="text-muted">${results.length} face(s) detected</small>
        `;
        container.appendChild(header);
        
        // Display each result, with best match highlighted
        results.forEach((result, index) => {
            const resultItem = this.createRealTimeResultItem(result, index === 0);
            container.appendChild(resultItem);
        });
    }

    createRealTimeResultItem(result, isBestMatch = false) {
        const item = document.createElement('div');
        item.className = `realtime-result-item ${isBestMatch ? 'best-match' : ''} fade-in`;
        
        const confidence = Math.round(result.confidence * 100);
        const confidenceClass = confidence > 80 ? 'success' : confidence > 60 ? 'warning' : 'danger';
        
        item.innerHTML = `
            <div class="result-card ${isBestMatch ? 'border-success' : ''}">
                ${isBestMatch ? '<div class="best-match-indicator"><i class="fas fa-crown"></i> Best Match</div>' : ''}
                <div class="result-content">
                    <div class="student-info">
                        <h6 class="student-name ${isBestMatch ? 'text-success font-weight-bold' : ''}">
                            ${result.name}
                        </h6>
                        <div class="student-details">
                            <small><i class="fas fa-id-card"></i> ${result.student_id}</small><br>
                            <small><i class="fas fa-graduation-cap"></i> ${result.department}</small><br>
                            <small><i class="fas fa-calendar"></i> Year ${result.year}</small>
                        </div>
                    </div>
                    <div class="confidence-display">
                        <div class="confidence-bar">
                            <div class="confidence-fill bg-${confidenceClass}" 
                                 style="width: ${confidence}%"></div>
                        </div>
                        <span class="badge badge-${confidenceClass}">${confidence}%</span>
                    </div>
                    <div class="action-section">
                        <button class="btn btn-${isBestMatch ? 'success' : 'outline-success'} btn-sm w-100" 
                                onclick="markAttendance('${result.student_id}', '${result.name}')">
                            <i class="fas fa-check"></i> Mark Attendance
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        return item;
    }

    createResultItem(result, isRealTime = false) {
        const item = document.createElement('div');
        item.className = 'result-item fade-in';
        
        const confidence = Math.round(result.confidence * 100);
        const confidenceClass = confidence > 80 ? 'success' : confidence > 60 ? 'warning' : 'error';
        
        item.innerHTML = `
            <div class="result-info">
                <div class="result-name">${result.name}</div>
                <div class="result-details">
                    ID: ${result.student_id} | 
                    ${result.department} | 
                    Year ${result.year}
                </div>
            </div>
            <div class="d-flex align-center gap-2">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%; background-color: var(--${confidenceClass}-color);"></div>
                </div>
                <span class="badge badge-${confidenceClass}">${confidence}%</span>
                ${isRealTime ? `<button class="btn btn-success btn-sm" onclick="markAttendance('${result.student_id}')">Mark Attendance</button>` : ''}
            </div>
        `;
        
        return item;
    }

    createEnhancedResultItem(result, isBestMatch = false) {
        const item = document.createElement('div');
        item.className = `enhanced-result-item ${isBestMatch ? 'best-match' : ''}`;
        
        const confidence = Math.round(result.confidence * 100);
        const confidenceClass = confidence > 80 ? 'success' : confidence > 60 ? 'warning' : 'error';
        
        item.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    ${isBestMatch ? '<span class="best-match-badge"><i class="fas fa-crown"></i> Best Match</span>' : ''}
                    <div class="student-photo-container">
                        <img src="/student_photo/${result.student_id}" 
                             alt="${result.name}" 
                             class="student-photo-result"
                             onerror="this.src='/static/images/default-avatar.svg'">
                    </div>
                </div>
                <div class="result-body">
                    <h5 class="student-name">${result.name}</h5>
                    <div class="student-details">
                        <div><i class="fas fa-id-card"></i> ${result.student_id}</div>
                        <div><i class="fas fa-graduation-cap"></i> ${result.department}</div>
                        <div><i class="fas fa-calendar"></i> Year ${result.year}</div>
                    </div>
                    <div class="confidence-section">
                        <div class="confidence-label">Match Confidence:</div>
                        <div class="confidence-display">
                            <div class="confidence-bar-large">
                                <div class="confidence-fill confidence-${confidenceClass}" 
                                     style="width: ${confidence}%"></div>
                            </div>
                            <span class="confidence-percentage badge badge-${confidenceClass}">${confidence}%</span>
                        </div>
                    </div>
                    <div class="action-buttons">
                        <button class="btn btn-success" onclick="app.markAttendanceForStudent('${result.student_id}', '${result.name}')">
                            <i class="fas fa-check"></i> Mark Attendance
                        </button>
                        <button class="btn btn-outline-primary" onclick="app.viewStudentDetails('${result.student_id}')">
                            <i class="fas fa-eye"></i> View Details
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        return item;
    }

    showAddNewUserDialog() {
        // Create modal for adding new user
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h4><i class="fas fa-user-plus"></i> Register New Student</h4>
                        <button class="btn-close" onclick="this.closest('.modal-overlay').remove()">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="modal-body">
                        <p>The captured face doesn't match any registered students.</p>
                        <p>Would you like to register this person as a new student?</p>
                        <div class="captured-preview" id="capturedPreview">
                            <!-- The captured image will be displayed here -->
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-primary" onclick="app.redirectToAddStudent()">
                            <i class="fas fa-user-plus"></i> Add New Student
                        </button>
                        <button class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove()">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Display the captured image if available
        if (this.canvas && this.canvas.toDataURL) {
            const preview = modal.querySelector('#capturedPreview');
            preview.innerHTML = `
                <img src="${this.canvas.toDataURL('image/jpeg', 0.8)}" 
                     alt="Captured face" 
                     class="captured-face-preview">
            `;
        }
    }

    redirectToAddStudent() {
        window.location.href = '/add_student';
    }

    resetUploadForm() {
        // Reset the photo recognition form
        const form = document.getElementById('photoRecognitionForm');
        const fileInput = document.getElementById('uploadPhoto');
        const wrapper = document.querySelector('#photoRecognitionForm .file-input-wrapper');
        const resultsContainer = document.getElementById('recognitionResults');

        if (form) form.reset();

        if (wrapper) {
            // Reset visual feedback
            wrapper.style.borderColor = '';
            wrapper.style.backgroundColor = '';
            const iconElement = wrapper.querySelector('i');
            if (iconElement) {
                iconElement.className = 'fas fa-upload';
            }
            // Reset text
            const textNode = wrapper.childNodes[wrapper.childNodes.length - 1];
            if (textNode && textNode.nodeType === Node.TEXT_NODE) {
                textNode.textContent = 'Select photo for recognition';
            }
        }

        if (resultsContainer) {
            resultsContainer.innerHTML = '<p class="text-center text-muted">Upload a photo to see recognition results</p>';
        }

        this.showAlert('Form reset. You can now upload a new image.', 'info', 3000);
    }

    markAttendanceForStudent(studentId, studentName) {
        if (confirm(`Mark attendance for ${studentName} (${studentId})?`)) {
            this.markAttendance(studentId);
        }
    }

    viewStudentDetails(studentId) {
        window.location.href = `/edit_student/${studentId}`;
    }

    clearRecognitionResults() {
        const containers = ['recognitionResults', 'realTimeResults'];
        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                if (id === 'realTimeResults') {
                    container.innerHTML = `
                        <div class="no-results">
                            <p class="text-center text-muted">
                                <i class="fas fa-info-circle"></i> Real-time recognition stopped
                            </p>
                        </div>
                    `;
                } else {
                    container.innerHTML = '';
                }
            }
        });
    }

    updateCameraControls(isActive) {
        const startBtn = document.getElementById('startCamera');
        const stopBtn = document.getElementById('stopCamera');
        const captureBtn = document.getElementById('capturePhoto');
        const toggleRecognitionBtn = document.getElementById('toggleRecognition');
        
        // Update button visibility
        if (startBtn) {
            startBtn.style.display = isActive ? 'none' : 'inline-flex';
            startBtn.classList.toggle('disabled', isActive);
        }
        
        if (stopBtn) {
            stopBtn.style.display = isActive ? 'inline-flex' : 'none';
            stopBtn.classList.toggle('disabled', !isActive);
        }
        
        if (captureBtn) {
            captureBtn.disabled = !isActive;
            captureBtn.classList.toggle('disabled', !isActive);
        }
        
        if (toggleRecognitionBtn) {
            toggleRecognitionBtn.disabled = !isActive;
            toggleRecognitionBtn.classList.toggle('disabled', !isActive);
        }
        
        // Update video element status
        if (this.video) {
            if (isActive) {
                this.video.classList.add('active');
                this.video.classList.remove('inactive');
            } else {
                this.video.classList.add('inactive');
                this.video.classList.remove('active');
            }
        }
        
        // Log current state for debugging
        console.log(`Camera controls updated: isActive=${isActive}`);
    }

    updateRecognitionButton() {
        const btn = document.getElementById('toggleRecognition');
        if (!btn) return;

        if (this.isRecognitionActive) {
            btn.innerHTML = '<i class="fas fa-stop recognition-active"></i> Stop Recognition';
            btn.className = 'btn btn-danger';
            btn.classList.add('recognition-active');
        } else {
            btn.innerHTML = '<i class="fas fa-eye"></i> Start Recognition';
            btn.className = 'btn btn-success';
            btn.classList.remove('recognition-active');
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) {
            this.removeImagePreview(event.target);
            return;
        }

        // Comprehensive file validation
        const validationResult = this.validateImageFile(file);
        if (!validationResult.valid) {
            this.showAlert(validationResult.message, 'error');
            event.target.value = '';
            this.removeImagePreview(event.target);
            return;
        }

        // Show success message for valid file
        this.showAlert(`Image "${file.name}" selected successfully`, 'success', 3000);

        // Show preview if it's an image
        this.showImagePreview(file, event.target);
    }

    validateImageFile(file) {
        // Check file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
        if (!allowedTypes.includes(file.type.toLowerCase())) {
            return {
                valid: false,
                message: 'Invalid file type. Please select a valid image file (JPG, PNG, GIF, BMP, WebP)'
            };
        }

        // Check file size (max 16MB)
        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            return {
                valid: false,
                message: `File size too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum allowed size is 16MB.`
            };
        }

        // Check minimum file size (avoid empty files)
        if (file.size < 1024) { // 1KB minimum
            return {
                valid: false,
                message: 'File appears to be too small or corrupted. Please select a valid image file.'
            };
        }

        // Additional checks for image dimensions could be added here
        return { valid: true, message: 'File is valid' };
    }

    removeImagePreview(input) {
        const preview = input.parentElement.querySelector('.image-preview');
        if (preview) {
            preview.remove();
        }
    }

    showImagePreview(file, input) {
        const reader = new FileReader();
        reader.onload = (e) => {
            // Create or update preview
            let preview = input.parentElement.querySelector('.image-preview');
            if (!preview) {
                preview = document.createElement('div');
                preview.className = 'image-preview mt-2';
                input.parentElement.appendChild(preview);
            }
            
            preview.innerHTML = `
                <img src="${e.target.result}" alt="Preview" style="max-width: 200px; max-height: 200px; border-radius: 8px; box-shadow: var(--shadow);">
                <p class="mt-1">${file.name}</p>
            `;
        };
        reader.readAsDataURL(file);
    }

    handleFormSubmit(event) {
        const form = event.target;
        const submitButton = form.querySelector('button[type="submit"]');
        
        // Validate required fields before submission
        const requiredFields = form.querySelectorAll('[required]');
        let hasErrors = false;
        
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                this.showFieldError(field, 'This field is required');
                hasErrors = true;
            } else {
                this.clearFieldError(field);
            }
        });
        
        // Validate email format if email field exists
        const emailField = form.querySelector('input[type="email"]');
        if (emailField && emailField.value) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(emailField.value)) {
                this.showFieldError(emailField, 'Please enter a valid email address');
                hasErrors = true;
            }
        }
        
        if (hasErrors) {
            event.preventDefault();
            this.showAlert('Please fix the errors in the form before submitting', 'error');
            return;
        }
        
        if (submitButton) {
            this.showLoading(submitButton.id || 'submit');
            submitButton.disabled = true;
            
            // Re-enable button after timeout to prevent permanent disable
            setTimeout(() => {
                submitButton.disabled = false;
                this.hideLoading(submitButton.id || 'submit');
            }, 30000); // 30 seconds timeout
        }
    }

    showFieldError(field, message) {
        this.clearFieldError(field);
        const errorElement = document.createElement('div');
        errorElement.className = 'field-error';
        errorElement.style.color = 'var(--error-color)';
        errorElement.style.fontSize = '0.875rem';
        errorElement.style.marginTop = '0.25rem';
        errorElement.textContent = message;
        field.parentElement.appendChild(errorElement);
        field.style.borderColor = 'var(--error-color)';
    }

    clearFieldError(field) {
        const existingError = field.parentElement.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }
        field.style.borderColor = '';
    }

    confirmDelete(event) {
        event.preventDefault();
        
        // Find the actual link element (could be event.target or its parent)
        let linkElement = event.target;
        
        // If we clicked on an icon inside the link, get the parent link
        if (linkElement.tagName === 'I') {
            linkElement = linkElement.parentElement;
        }
        
        // Get the confirmation message and href from the link element
        const message = linkElement.dataset.confirm || 'Are you sure you want to delete this item?';
        const href = linkElement.href;
        
        if (confirm(message)) {
            window.location.href = href;
        }
    }

    showAlert(message, type = 'info', timeout = 5000) {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.dynamic-alert');
        existingAlerts.forEach(alert => alert.remove());
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} dynamic-alert fade-in`;
        alert.innerHTML = `
            <span>${message}</span>
            <button type="button" class="close-alert" style="float: right; background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>
        `;
        
        // Add to container or body
        const container = document.querySelector('.container') || document.body;
        container.insertBefore(alert, container.firstChild);
        
        // Add close functionality
        alert.querySelector('.close-alert').addEventListener('click', () => {
            alert.remove();
        });
        
        // Auto-hide after specified timeout
        if (timeout > 0) {
            setTimeout(() => {
                if (alert.parentElement) {
                    alert.remove();
                }
            }, timeout);
        }
    }

    showLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.disabled = true;
            const originalText = element.textContent;
            element.dataset.originalText = originalText;
            element.innerHTML = '<span class="loading"></span> Loading...';
        }
    }

    hideLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.disabled = false;
            const originalText = element.dataset.originalText || 'Submit';
            element.textContent = originalText;
        }
    }

    autoHideAlerts() {
        const alerts = document.querySelectorAll('.alert:not(.dynamic-alert)');
        alerts.forEach(alert => {
            setTimeout(() => {
                if (alert.parentElement) {
                    alert.style.opacity = '0';
                    setTimeout(() => alert.remove(), 300);
                }
            }, 5000);
        });
    }

    initializeTooltips() {
        // Simple tooltip implementation
        const elements = document.querySelectorAll('[data-tooltip]');
        elements.forEach(element => {
            element.addEventListener('mouseenter', this.showTooltip.bind(this));
            element.addEventListener('mouseleave', this.hideTooltip.bind(this));
        });
    }

    showTooltip(event) {
        const element = event.target;
        const text = element.dataset.tooltip;
        
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = text;
        tooltip.style.cssText = `
            position: absolute;
            background: #333;
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.875rem;
            z-index: 1000;
            pointer-events: none;
        `;
        
        document.body.appendChild(tooltip);
        
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + 'px';
        tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
        
        element._tooltip = tooltip;
    }

    hideTooltip(event) {
        const element = event.target;
        if (element._tooltip) {
            element._tooltip.remove();
            delete element._tooltip;
        }
    }
}

// Global functions for inline event handlers
async function markAttendance(studentId, studentName = null) {
    try {
        const response = await fetch('/mark_attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ student_id: studentId })
        });

        const result = await response.json();
        
        if (result.success) {
            const displayName = studentName || studentId;
            app.showAlert(`✅ Attendance marked for ${displayName}`, 'success');
            
            // Update UI to show attendance marked
            updateAttendanceButton(studentId, true);
        } else {
            app.showAlert(result.message || 'Failed to mark attendance', 'error');
        }
    } catch (error) {
        console.error('Error marking attendance:', error);
        app.showAlert('Failed to mark attendance', 'error');
    }
}

function updateAttendanceButton(studentId, isMarked) {
    // Find and update the attendance button for this student
    const buttons = document.querySelectorAll(`button[onclick*="${studentId}"]`);
    buttons.forEach(button => {
        if (button.textContent.includes('Mark Attendance')) {
            if (isMarked) {
                button.innerHTML = '<i class="fas fa-check-circle"></i> Marked';
                button.className = button.className.replace('btn-success', 'btn-secondary');
                button.disabled = true;
            }
        }
    });
}

async function bulkMarkAttendance() {
    const fileInput = document.getElementById('groupPhoto');
    if (!fileInput.files[0]) {
        app.showAlert('Please select a group photo first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('photo', fileInput.files[0]);
    
    // Check if location extraction is enabled
    const extractLocationCheckbox = document.getElementById('extractLocation');
    const extractLocation = extractLocationCheckbox ? extractLocationCheckbox.checked : true;
    formData.append('extract_location', extractLocation ? 'true' : 'false');

    try {
        app.showLoading('bulkMarkBtn');
        
        const response = await fetch('/bulk_attendance', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.success) {
            const data = result.data;
            let message = `Processed ${data.total_faces} faces. `;
            message += `Marked: ${data.marked_attendance.length}, `;
            message += `Already marked: ${data.already_marked.length}, `;
            message += `Errors: ${data.errors.length}`;
            
            // Add location information to message if available
            if (data.location_data && data.location_data.has_location) {
                message += ` | Location: ${data.location_data.address}`;
            } else if (extractLocation) {
                message += ` | No location data found`;
            }
            
            app.showAlert(message, 'success');
            
            // Display detailed results
            displayBulkResults(data);
        } else {
            app.showAlert(result.message || 'Failed to process group photo', 'error');
        }
    } catch (error) {
        console.error('Error processing group photo:', error);
        app.showAlert('Failed to process group photo', 'error');
    } finally {
        app.hideLoading('bulkMarkBtn');
    }
}

function displayBulkResults(data) {
    const container = document.getElementById('bulkResults');
    if (!container) return;

    let html = '<div class="bulk-results-container">';
    html += '<h4><i class="fas fa-chart-bar"></i> Bulk Attendance Results</h4>';
    
    // Location information section
    if (data.location_data) {
        html += '<div class="location-info-section mb-3">';
        html += '<h5><i class="fas fa-map-marker-alt"></i> Location Information</h5>';
        
        if (data.location_data.has_location) {
            html += '<div class="alert alert-info location-alert">';
            html += `<strong>Location:</strong> ${data.location_data.address}<br>`;
            html += `<strong>Coordinates:</strong> ${data.location_data.latitude.toFixed(6)}, ${data.location_data.longitude.toFixed(6)}`;
            
            if (data.location_data.city || data.location_data.state || data.location_data.country) {
                html += '<br><strong>Details:</strong> ';
                const details = [data.location_data.city, data.location_data.state, data.location_data.country]
                    .filter(detail => detail && detail.trim()).join(', ');
                html += details;
            }
            html += '</div>';
        } else {
            html += '<div class="alert alert-warning">';
            html += '<i class="fas fa-exclamation-triangle"></i> No location data found in the uploaded image.';
            html += '</div>';
        }
        html += '</div>';
    }
    
    // Attendance results section
    html += '<div class="attendance-results-section">';
    
    if (data.marked_attendance.length > 0) {
        html += '<div class="result-section success-section">';
        html += '<h5 class="text-success"><i class="fas fa-check-circle"></i> Successfully Marked Attendance:</h5>';
        html += '<div class="result-list">';
        data.marked_attendance.forEach(item => {
            html += '<div class="result-item success-item">';
            html += `<div class="student-info">`;
            html += `<strong>${item.name}</strong> (${item.student_id})`;
            html += `<span class="confidence-badge">${Math.round(item.confidence * 100)}%</span>`;
            html += `</div>`;
            
            // Add location info for individual attendance if available
            if (item.location && item.coordinates) {
                html += `<div class="location-details">`;
                html += `<small><i class="fas fa-map-pin"></i> ${item.location} (${item.coordinates})</small>`;
                html += `</div>`;
            }
            html += '</div>';
        });
        html += '</div>';
        html += '</div>';
    }
    
    if (data.already_marked.length > 0) {
        html += '<div class="result-section warning-section">';
        html += '<h5 class="text-warning"><i class="fas fa-clock"></i> Already Marked Today:</h5>';
        html += '<div class="result-list">';
        data.already_marked.forEach(item => {
            html += '<div class="result-item warning-item">';
            html += `<strong>${item.name}</strong> (${item.student_id})`;
            html += '</div>';
        });
        html += '</div>';
        html += '</div>';
    }
    
    if (data.errors.length > 0) {
        html += '<div class="result-section error-section">';
        html += '<h5 class="text-danger"><i class="fas fa-exclamation-circle"></i> Errors:</h5>';
        html += '<div class="result-list">';
        data.errors.forEach(item => {
            html += '<div class="result-item error-item">';
            html += `<span class="error-message">${item.error}</span>`;
            html += '</div>';
        });
        html += '</div>';
        html += '</div>';
    }
    
    html += '</div>'; // attendance-results-section
    html += '</div>'; // bulk-results-container
    
    container.innerHTML = html;
}

// Initialize the application when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new FaceRecognitionApp();
});

// Face Comparison and Verification Functions
let selectedMatch = null;

async function confirmMatch() {
    if (!selectedMatch) {
        app.showAlert('Please select a match first', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/mark_attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                student_id: selectedMatch.student_id,
                verified: true,
                similarity_score: selectedMatch.similarity_score 
            })
        });

        const result = await response.json();
        
        if (result.success) {
            showVerificationFeedback(`✅ Attendance marked for ${selectedMatch.student_name}`, 'success');
            
            // Clear current selection and continue scanning
            clearVerificationState();
        } else {
            showVerificationFeedback(result.message || 'Failed to mark attendance', 'error');
        }
    } catch (error) {
        console.error('Error confirming match:', error);
        showVerificationFeedback('Failed to mark attendance', 'error');
    }
}

function skipMatch() {
    showVerificationFeedback('Face skipped. Continuing scan...', 'warning');
    clearVerificationState();
}

function rejectAllMatches() {
    showVerificationFeedback('No match confirmed. Face not recognized.', 'warning');
    clearVerificationState();
}

function clearVerificationState() {
    selectedMatch = null;
    
    // Clear selection
    document.querySelectorAll('.match-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Hide verification controls temporarily
    const controls = document.getElementById('verificationControls');
    setTimeout(() => {
        if (controls) controls.style.display = 'none';
    }, 2000);
    
    // Clear face display
    setTimeout(() => {
        clearFaceDisplay();
    }, 3000);
}

function showVerificationFeedback(message, type) {
    const feedback = document.getElementById('verificationFeedback');
    if (feedback) {
        feedback.textContent = message;
        feedback.className = `verification-feedback ${type}`;
        feedback.style.display = 'block';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            feedback.style.display = 'none';
        }, 3000);
    }
}

function clearFaceDisplay() {
    const image = document.getElementById('realtimeFaceImage');
    const container = document.getElementById('realtimeFaceContainer');
    
    if (image && container) {
        image.style.display = 'none';
        const placeholder = container.querySelector('.no-face-placeholder');
        if (placeholder) {
            placeholder.style.display = 'block';
        }
        
        // Clear matches
        const matchesContainer = document.getElementById('suggestedMatches');
        if (matchesContainer) {
            matchesContainer.innerHTML = `
                <div class="no-matches-placeholder">
                    <i class="fas fa-search"></i>
                    <p>Scanning for faces...</p>
                </div>
            `;
        }
        
        // Hide verification controls
        const controls = document.getElementById('verificationControls');
        if (controls) {
            controls.style.display = 'none';
        }
    }
}

function selectMatch(match, element) {
    // Remove previous selections
    document.querySelectorAll('.match-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Select current match
    element.classList.add('selected');
    selectedMatch = match;
    
    // Update verification controls
    const confirmBtn = document.getElementById('confirmMatchBtn');
    if (confirmBtn) {
        confirmBtn.disabled = false;
    }
}

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceRecognitionApp;
}
// ========================================
// FUTURISTIC LOGIN PAGE - INTERACTIVE ANIMATIONS
// Next-Gen Face Recognition Attendance System
// ========================================

class FuturisticLogin {
    constructor() {
        this.particles = [];
        this.mouse = { x: 0, y: 0 };
        this.isPasswordVisible = false;
        this.isLoading = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initParticleSystem();
        this.initMouseTracking();
        this.initFormInteractions();
        this.initPasswordToggle();
        this.initLoginButton();
        this.setupFormValidation();
        
        // Initialize 3D effects
        this.init3DEffects();
        
        console.log('🚀 FuturisticLogin System Initialized');
    }

    // ========================================
    // PARTICLE SYSTEM
    // ========================================
    
    initParticleSystem() {
        this.particleField = document.getElementById('particleField');
        this.createParticles();
        this.animateParticles();
    }

    createParticles() {
        const particleCount = window.innerWidth < 768 ? 30 : 50;
        
        for (let i = 0; i < particleCount; i++) {
            const particle = this.createParticle();
            this.particles.push(particle);
            this.particleField.appendChild(particle.element);
        }
    }

    createParticle() {
        const element = document.createElement('div');
        element.className = 'particle';
        
        const size = Math.random() * 4 + 1;
        const x = Math.random() * window.innerWidth;
        const y = Math.random() * window.innerHeight;
        const speedX = (Math.random() - 0.5) * 0.5;
        const speedY = (Math.random() - 0.5) * 0.5;
        const opacity = Math.random() * 0.5 + 0.2;
        
        // Random neon colors
        const colors = ['#2575fc', '#00c9ff', '#ff0080', '#00ff88', '#ffff00'];
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        element.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            border-radius: 50%;
            opacity: ${opacity};
            box-shadow: 0 0 ${size * 3}px ${color};
            pointer-events: none;
            transform: translate(${x}px, ${y}px);
            transition: all 0.3s ease;
        `;

        return {
            element,
            x,
            y,
            speedX,
            speedY,
            size,
            color,
            originalOpacity: opacity
        };
    }

    animateParticles() {
        this.particles.forEach(particle => {
            // Move particles
            particle.x += particle.speedX;
            particle.y += particle.speedY;

            // Wrap around screen
            if (particle.x > window.innerWidth) particle.x = 0;
            if (particle.x < 0) particle.x = window.innerWidth;
            if (particle.y > window.innerHeight) particle.y = 0;
            if (particle.y < 0) particle.y = window.innerHeight;

            // Mouse interaction
            const dx = this.mouse.x - particle.x;
            const dy = this.mouse.y - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const maxDistance = 150;

            if (distance < maxDistance) {
                const force = (maxDistance - distance) / maxDistance;
                const opacity = particle.originalOpacity + force * 0.5;
                const scale = 1 + force * 2;
                
                particle.element.style.opacity = Math.min(opacity, 1);
                particle.element.style.transform = `translate(${particle.x}px, ${particle.y}px) scale(${scale})`;
            } else {
                particle.element.style.opacity = particle.originalOpacity;
                particle.element.style.transform = `translate(${particle.x}px, ${particle.y}px) scale(1)`;
            }
        });

        requestAnimationFrame(() => this.animateParticles());
    }

    // ========================================
    // MOUSE TRACKING & 3D EFFECTS
    // ========================================
    
    initMouseTracking() {
        document.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
            
            this.updateMouseEffects(e);
        });
    }

    updateMouseEffects(e) {
        const card = document.getElementById('loginCard');
        const rect = card.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        const deltaX = (e.clientX - centerX) / rect.width;
        const deltaY = (e.clientY - centerY) / rect.height;
        
        // 3D tilt effect
        const tiltX = deltaY * 10;
        const tiltY = deltaX * -10;
        
        card.style.transform = `
            translateY(-10px) 
            rotateX(${tiltX}deg) 
            rotateY(${tiltY}deg)
            perspective(1000px)
        `;

        // Update gradient overlay based on mouse position
        const gradientOverlay = document.querySelector('.gradient-overlay');
        if (gradientOverlay) {
            const x = (e.clientX / window.innerWidth) * 100;
            const y = (e.clientY / window.innerHeight) * 100;
            
            gradientOverlay.style.background = `
                radial-gradient(circle at ${x}% ${y}%, #6a11cb 0%, transparent 50%),
                radial-gradient(circle at ${100-x}% ${100-y}%, #2575fc 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, #00c9ff 0%, transparent 50%),
                linear-gradient(135deg, #050505 0%, #0a0a0a 100%)
            `;
        }
    }

    init3DEffects() {
        // Reset card transform when mouse leaves
        document.addEventListener('mouseleave', () => {
            const card = document.getElementById('loginCard');
            card.style.transform = 'translateY(-10px) rotateX(0deg) rotateY(0deg)';
        });

        // Floating spheres interaction
        this.initSphereInteraction();
    }

    initSphereInteraction() {
        const spheres = document.querySelectorAll('.sphere');
        
        document.addEventListener('mousemove', (e) => {
            spheres.forEach((sphere, index) => {
                const speed = 0.5 + index * 0.1;
                const x = (e.clientX * speed) / 100;
                const y = (e.clientY * speed) / 100;
                
                sphere.style.transform = `
                    translate(${x}px, ${y}px) 
                    rotateZ(${x * 0.1}deg) 
                    scale(${1 + Math.sin(Date.now() * 0.001 + index) * 0.1})
                `;
            });
        });
    }

    // ========================================
    // PASSWORD TOGGLE WITH EYE ANIMATION
    // ========================================
    
    initPasswordToggle() {
        const passwordToggle = document.getElementById('passwordToggle');
        const passwordInput = document.getElementById('password');
        const eyeIcon = document.getElementById('eyeIcon');
        
        passwordToggle.addEventListener('click', () => {
            this.togglePasswordVisibility(passwordInput, eyeIcon);
        });

        // Eye blink animation on hover
        passwordToggle.addEventListener('mouseenter', () => {
            this.triggerEyeBlink(eyeIcon);
        });
    }

    togglePasswordVisibility(passwordInput, eyeIcon) {
        this.isPasswordVisible = !this.isPasswordVisible;
        
        // Animate the eye
        this.animateEyeToggle(eyeIcon);
        
        // Toggle password visibility
        passwordInput.type = this.isPasswordVisible ? 'text' : 'password';
        eyeIcon.className = this.isPasswordVisible ? 'fas fa-eye-slash eye-icon' : 'fas fa-eye eye-icon';
        
        // Create ripple effect
        this.createRippleEffect(document.querySelector('.password-toggle'));
    }

    animateEyeToggle(eyeIcon) {
        // Blink animation
        eyeIcon.style.transform = 'scaleY(0.1)';
        eyeIcon.style.filter = 'brightness(2)';
        
        setTimeout(() => {
            eyeIcon.style.transform = 'scaleY(1)';
            eyeIcon.style.filter = 'brightness(1)';
        }, 150);

        // Color pulse
        this.pulseEyeColor(eyeIcon);
    }

    triggerEyeBlink(eyeIcon) {
        eyeIcon.style.animation = 'none';
        setTimeout(() => {
            eyeIcon.style.animation = 'eyeBlink 0.3s ease-in-out';
        }, 10);
    }

    pulseEyeColor(eyeIcon) {
        const colors = ['#2575fc', '#00c9ff', '#ff0080', '#00ff88'];
        let colorIndex = 0;
        
        const colorInterval = setInterval(() => {
            eyeIcon.style.color = colors[colorIndex];
            colorIndex = (colorIndex + 1) % colors.length;
        }, 100);
        
        setTimeout(() => {
            clearInterval(colorInterval);
            eyeIcon.style.color = '#2575fc';
        }, 500);
    }

    // ========================================
    // FORM INTERACTIONS
    // ========================================
    
    initFormInteractions() {
        const inputs = document.querySelectorAll('.futuristic-input');
        
        inputs.forEach(input => {
            this.setupInputEffects(input);
        });
    }

    setupInputEffects(input) {
        // Focus effects
        input.addEventListener('focus', () => {
            this.activateInputGlow(input);
            this.createInputParticles(input);
        });

        // Blur effects
        input.addEventListener('blur', () => {
            this.deactivateInputGlow(input);
        });

        // Typing effects
        input.addEventListener('input', () => {
            this.createTypingEffect(input);
        });
    }

    activateInputGlow(input) {
        const container = input.closest('.input-container');
        const glow = container.querySelector('.input-glow');
        
        glow.style.opacity = '0.4';
        glow.style.transform = 'scale(1.02)';
        
        // Sound effect simulation (visual pulse)
        this.createSoundVisualization(container);
    }

    deactivateInputGlow(input) {
        const container = input.closest('.input-container');
        const glow = container.querySelector('.input-glow');
        
        glow.style.opacity = '0';
        glow.style.transform = 'scale(1)';
    }

    createInputParticles(input) {
        const container = input.closest('.input-container');
        const rect = container.getBoundingClientRect();
        
        for (let i = 0; i < 5; i++) {
            this.createTempParticle(rect.left + rect.width * Math.random(), rect.top);
        }
    }

    createTempParticle(x, y) {
        const particle = document.createElement('div');
        particle.style.cssText = `
            position: fixed;
            width: 3px;
            height: 3px;
            background: #00c9ff;
            border-radius: 50%;
            box-shadow: 0 0 10px #00c9ff;
            pointer-events: none;
            z-index: 1000;
            left: ${x}px;
            top: ${y}px;
            animation: particleFloat 1s ease-out forwards;
        `;
        
        document.body.appendChild(particle);
        
        setTimeout(() => {
            particle.remove();
        }, 1000);
    }

    createTypingEffect(input) {
        const container = input.closest('.input-container');
        container.style.transform = 'scale(1.01)';
        
        setTimeout(() => {
            container.style.transform = 'scale(1)';
        }, 100);
    }

    createSoundVisualization(container) {
        const visualizer = document.createElement('div');
        visualizer.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00c9ff, transparent);
            animation: soundWave 0.5s ease-out;
            pointer-events: none;
        `;
        
        container.appendChild(visualizer);
        
        setTimeout(() => {
            visualizer.remove();
        }, 500);
    }

    // ========================================
    // LOGIN BUTTON INTERACTIONS
    // ========================================
    
    initLoginButton() {
        const loginButton = document.getElementById('loginButton');
        
        if (!loginButton) return; // Exit if login button doesn't exist
        
        loginButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.handleLogin();
        });

        // Hover effects
        loginButton.addEventListener('mouseenter', () => {
            this.activateButtonEffects(loginButton);
        });

        loginButton.addEventListener('mouseleave', () => {
            this.deactivateButtonEffects(loginButton);
        });
        
        // Fallback: Allow form submission if JavaScript fails
        const form = document.getElementById('loginForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                // If AJAX is working, prevent default submission
                if (this.isLoading) {
                    e.preventDefault();
                }
            });
        }
    }

    activateButtonEffects(button) {
        // Create orbital particles around button
        this.createOrbitalParticles(button);
        
        // Energy pulse effect
        this.createEnergyPulse(button);
    }

    deactivateButtonEffects(button) {
        // Remove orbital particles
        const orbitals = button.querySelectorAll('.orbital-particle');
        orbitals.forEach(orbital => orbital.remove());
    }

    createOrbitalParticles(button) {
        const rect = button.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        for (let i = 0; i < 6; i++) {
            const particle = document.createElement('div');
            particle.className = 'orbital-particle';
            particle.style.cssText = `
                position: absolute;
                width: 4px;
                height: 4px;
                background: #ff0080;
                border-radius: 50%;
                box-shadow: 0 0 8px #ff0080;
                left: ${centerX}px;
                top: ${centerY}px;
                animation: orbit${i} 2s linear infinite;
                pointer-events: none;
            `;
            
            button.appendChild(particle);
        }
    }

    createEnergyPulse(button) {
        const pulse = document.createElement('div');
        pulse.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 0;
            height: 0;
            border: 2px solid #00c9ff;
            border-radius: 50%;
            animation: energyPulse 1s ease-out infinite;
            pointer-events: none;
        `;
        
        button.appendChild(pulse);
        
        setTimeout(() => {
            pulse.remove();
        }, 1000);
    }

    createRippleEffect(element) {
        const ripple = document.createElement('div');
        const rect = element.getBoundingClientRect();
        
        ripple.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 0;
            height: 0;
            background: rgba(0, 201, 255, 0.3);
            border-radius: 50%;
            animation: ripple 0.6s ease-out;
            pointer-events: none;
        `;
        
        element.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    }

    // ========================================
    // FORM VALIDATION & SUBMISSION
    // ========================================
    
    setupFormValidation() {
        const form = document.getElementById('loginForm');
        const inputs = form.querySelectorAll('.futuristic-input');
        
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateField(input);
            });
        });
    }

    validateField(input) {
        const value = input.value.trim();
        const container = input.closest('.input-container');
        
        // Remove existing validation indicators
        this.clearValidationState(container);
        
        if (value.length === 0) {
            this.showFieldError(container, 'This field is required');
        } else if (input.type === 'password' && value.length < 6) {
            this.showFieldError(container, 'Password must be at least 6 characters');
        } else {
            this.showFieldSuccess(container);
        }
    }

    clearValidationState(container) {
        container.classList.remove('field-error', 'field-success');
        const errorMsg = container.querySelector('.error-message');
        if (errorMsg) errorMsg.remove();
    }

    showFieldError(container, message) {
        container.classList.add('field-error');
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: absolute;
            bottom: -20px;
            left: 0;
            font-size: 0.8rem;
            color: #ff0080;
            animation: slideIn 0.3s ease-out;
        `;
        
        container.appendChild(errorDiv);
    }

    showFieldSuccess(container) {
        container.classList.add('field-success');
    }

    async handleLogin() {
        if (this.isLoading) return;
        
        console.log('🚀 Starting login process...');
        
        const form = document.getElementById('loginForm');
        const formData = new FormData(form);
        
        // Validate form
        if (!this.validateForm()) {
            console.log('❌ Form validation failed');
            this.showMessage('Please fill in all required fields', 'error');
            return;
        }
        
        console.log('✅ Form validation passed');
        this.showLoading(true);
        
        try {
            // Simulate API call delay for dramatic effect
            console.log('⏳ Adding dramatic delay...');
            await this.delay(1500);
            
            // Convert FormData to URLSearchParams for proper content type
            const params = new URLSearchParams();
            for (let [key, value] of formData) {
                params.append(key, value);
            }
            
            console.log('📡 Sending login request...');
            const response = await fetch(form.action, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: params
            });
            
            console.log('📨 Response received:', response.status);
            
            if (response.ok) {
                const result = await response.json();
                console.log('📋 Response data:', result);
                
                if (result.success) {
                    console.log('✅ Login successful, redirecting to:', result.redirect);
                    this.showMessage('Login successful! Redirecting...', 'success');
                    await this.delay(1000);
                    window.location.href = result.redirect || '/dashboard';
                } else {
                    console.log('❌ Login failed:', result.message);
                    this.showMessage(result.message || 'Login failed', 'error');
                }
            } else {
                console.log('❌ HTTP error:', response.status);
                try {
                    const result = await response.json();
                    this.showMessage(result.message || 'Invalid credentials. Please try again.', 'error');
                } catch (e) {
                    this.showMessage('Invalid credentials. Please try again.', 'error');
                }
            }
        } catch (error) {
            console.error('💥 Login error:', error);
            this.showMessage('Connection error. Please try again.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    validateForm() {
        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value.trim();
        
        return username.length > 0 && password.length >= 6;
    }

    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        this.isLoading = show;
        
        if (show) {
            overlay.classList.add('active');
            this.startLoadingAnimation();
        } else {
            overlay.classList.remove('active');
        }
    }

    startLoadingAnimation() {
        const loadingText = document.querySelector('.loading-text');
        const messages = [
            'Authenticating...',
            'Verifying credentials...',
            'Accessing system...',
            'Welcome!'
        ];
        
        let messageIndex = 0;
        const interval = setInterval(() => {
            if (!this.isLoading) {
                clearInterval(interval);
                return;
            }
            
            loadingText.textContent = messages[messageIndex];
            messageIndex = (messageIndex + 1) % messages.length;
        }, 500);
    }

    showMessage(text, type) {
        const container = document.getElementById('messageContainer');
        const textElement = container.querySelector('.message-text');
        const iconElement = container.querySelector('.message-icon');
        
        textElement.textContent = text;
        container.className = `message-container ${type}`;
        
        if (type === 'success') {
            iconElement.className = 'fas fa-check-circle message-icon';
        } else {
            iconElement.className = 'fas fa-exclamation-triangle message-icon';
        }
        
        container.classList.add('show');
        
        setTimeout(() => {
            container.classList.remove('show');
        }, 3000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // ========================================
    // EVENT LISTENERS SETUP
    // ========================================
    
    setupEventListeners() {
        // Window resize handler
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Performance optimization
        window.addEventListener('blur', () => {
            this.pauseAnimations();
        });

        window.addEventListener('focus', () => {
            this.resumeAnimations();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }

    handleResize() {
        // Recreate particles on resize
        this.particles.forEach(particle => {
            particle.element.remove();
        });
        this.particles = [];
        this.createParticles();
    }

    pauseAnimations() {
        document.body.style.animationPlayState = 'paused';
    }

    resumeAnimations() {
        document.body.style.animationPlayState = 'running';
    }

    handleKeyboardShortcuts(e) {
        // Enter key to submit
        if (e.key === 'Enter' && !e.shiftKey) {
            const activeElement = document.activeElement;
            if (activeElement.classList.contains('futuristic-input')) {
                e.preventDefault();
                this.handleLogin();
            }
        }
        
        // Escape to clear form
        if (e.key === 'Escape') {
            this.clearForm();
        }
    }

    clearForm() {
        const inputs = document.querySelectorAll('.futuristic-input');
        inputs.forEach(input => {
            input.value = '';
            this.clearValidationState(input.closest('.input-container'));
        });
    }
}

// ========================================
// CSS ANIMATIONS (Injected Dynamically)
// ========================================

function injectAdditionalStyles() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes particleFloat {
            0% { transform: translateY(0) scale(1); opacity: 1; }
            100% { transform: translateY(-50px) scale(0); opacity: 0; }
        }
        
        @keyframes soundWave {
            0% { transform: scaleX(0); opacity: 1; }
            100% { transform: scaleX(1); opacity: 0; }
        }
        
        @keyframes eyeBlink {
            0%, 100% { transform: scaleY(1); }
            50% { transform: scaleY(0.1); }
        }
        
        @keyframes ripple {
            0% { width: 0; height: 0; opacity: 1; }
            100% { width: 100px; height: 100px; opacity: 0; }
        }
        
        @keyframes energyPulse {
            0% { width: 0; height: 0; opacity: 1; }
            100% { width: 200px; height: 200px; opacity: 0; }
        }
        
        @keyframes slideIn {
            0% { transform: translateY(-10px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes orbit0 { 0% { transform: rotate(0deg) translateX(30px) rotate(0deg); } 100% { transform: rotate(360deg) translateX(30px) rotate(-360deg); } }
        @keyframes orbit1 { 0% { transform: rotate(60deg) translateX(30px) rotate(-60deg); } 100% { transform: rotate(420deg) translateX(30px) rotate(-420deg); } }
        @keyframes orbit2 { 0% { transform: rotate(120deg) translateX(30px) rotate(-120deg); } 100% { transform: rotate(480deg) translateX(30px) rotate(-480deg); } }
        @keyframes orbit3 { 0% { transform: rotate(180deg) translateX(30px) rotate(-180deg); } 100% { transform: rotate(540deg) translateX(30px) rotate(-540deg); } }
        @keyframes orbit4 { 0% { transform: rotate(240deg) translateX(30px) rotate(-240deg); } 100% { transform: rotate(600deg) translateX(30px) rotate(-600deg); } }
        @keyframes orbit5 { 0% { transform: rotate(300deg) translateX(30px) rotate(-300deg); } 100% { transform: rotate(660deg) translateX(30px) rotate(-660deg); } }
        
        .field-error .futuristic-input {
            border-color: #ff0080;
            box-shadow: 0 0 15px rgba(255, 0, 128, 0.3);
        }
        
        .field-success .futuristic-input {
            border-color: #00ff88;
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
        }
    `;
    document.head.appendChild(style);
}

// ========================================
// INITIALIZATION
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    // Only initialize FuturisticLogin on login page
    const loginCard = document.getElementById('loginCard');
    if (loginCard) {
        injectAdditionalStyles();
        new FuturisticLogin();
    }
});

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FuturisticLogin;
}
