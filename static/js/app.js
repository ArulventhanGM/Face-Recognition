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
        }

        // Auto-hide alerts
        this.autoHideAlerts();
        
        // Initialize tooltips if needed
        this.initializeTooltips();
    }

    async startCamera() {
        try {
            this.showLoading('startCamera');
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            if (this.video) {
                this.video.srcObject = this.stream;
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    this.updateCameraControls(true);
                    this.hideLoading('startCamera');
                    this.showAlert('Camera started successfully', 'success');
                };
            }
        } catch (error) {
            console.error('Error starting camera:', error);
            this.hideLoading('startCamera');
            this.showAlert('Failed to start camera. Please check permissions.', 'error');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.video) {
            this.video.srcObject = null;
        }

        this.stopRecognition();
        this.updateCameraControls(false);
        this.showAlert('Camera stopped', 'info');
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
                this.displayRecognitionResults(result.data);
                this.showAlert(`Recognized ${result.data.length} face(s)`, 'success');
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

    displayRecognitionResults(results) {
        const container = document.getElementById('recognitionResults');
        if (!container) return;

        container.innerHTML = '';
        
        if (results.length === 0) {
            container.innerHTML = '<p class="text-center">No faces recognized</p>';
            return;
        }

        results.forEach(result => {
            const resultItem = this.createResultItem(result);
            container.appendChild(resultItem);
        });
    }

    displayRealTimeResults(results) {
        const container = document.getElementById('realTimeResults');
        if (!container) return;

        container.innerHTML = '';
        
        results.forEach(result => {
            const resultItem = this.createResultItem(result, true);
            container.appendChild(resultItem);
        });
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

    clearRecognitionResults() {
        const containers = ['recognitionResults', 'realTimeResults'];
        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = '';
            }
        });
    }

    updateCameraControls(isActive) {
        const startBtn = document.getElementById('startCamera');
        const stopBtn = document.getElementById('stopCamera');
        const captureBtn = document.getElementById('capturePhoto');
        
        if (startBtn) startBtn.style.display = isActive ? 'none' : 'inline-flex';
        if (stopBtn) stopBtn.style.display = isActive ? 'inline-flex' : 'none';
        if (captureBtn) captureBtn.disabled = !isActive;
    }

    updateRecognitionButton() {
        const btn = document.getElementById('toggleRecognition');
        if (!btn) return;

        if (this.isRecognitionActive) {
            btn.textContent = 'Stop Recognition';
            btn.className = 'btn btn-danger';
        } else {
            btn.textContent = 'Start Recognition';
            btn.className = 'btn btn-success';
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showAlert('Please select a valid image file', 'error');
            event.target.value = '';
            return;
        }

        // Validate file size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            this.showAlert('File size too large. Maximum 16MB allowed.', 'error');
            event.target.value = '';
            return;
        }

        // Show preview if it's an image
        this.showImagePreview(file, event.target);
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
        
        if (submitButton) {
            this.showLoading(submitButton.id || 'submit');
        }
    }

    confirmDelete(event) {
        event.preventDefault();
        const message = event.target.dataset.confirm || 'Are you sure you want to delete this item?';
        
        if (confirm(message)) {
            window.location.href = event.target.href;
        }
    }

    showAlert(message, type = 'info') {
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
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
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
async function markAttendance(studentId) {
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
            app.showAlert(`Attendance marked for ${studentId}`, 'success');
        } else {
            app.showAlert(result.message || 'Failed to mark attendance', 'error');
        }
    } catch (error) {
        console.error('Error marking attendance:', error);
        app.showAlert('Failed to mark attendance', 'error');
    }
}

async function bulkMarkAttendance() {
    const fileInput = document.getElementById('groupPhoto');
    if (!fileInput.files[0]) {
        app.showAlert('Please select a group photo first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('photo', fileInput.files[0]);

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

    let html = '<h4>Bulk Attendance Results</h4>';
    
    if (data.marked_attendance.length > 0) {
        html += '<h5 class="text-success">Successfully Marked:</h5><ul>';
        data.marked_attendance.forEach(item => {
            html += `<li>${item.name} (${item.student_id}) - ${Math.round(item.confidence * 100)}%</li>`;
        });
        html += '</ul>';
    }
    
    if (data.already_marked.length > 0) {
        html += '<h5 class="text-warning">Already Marked:</h5><ul>';
        data.already_marked.forEach(item => {
            html += `<li>${item.name} (${item.student_id})</li>`;
        });
        html += '</ul>';
    }
    
    if (data.errors.length > 0) {
        html += '<h5 class="text-danger">Errors:</h5><ul>';
        data.errors.forEach(item => {
            html += `<li>${item.error}</li>`;
        });
        html += '</ul>';
    }
    
    container.innerHTML = html;
}

// Initialize the application when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new FaceRecognitionApp();
});

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceRecognitionApp;
}
