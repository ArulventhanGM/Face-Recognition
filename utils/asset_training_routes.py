#!/usr/bin/env python3
"""
Asset Training Routes for Face Recognition System
Provides Flask routes for training the face recognition model with asset images
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required
import os
import logging
from .asset_face_training import AssetFaceTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
asset_training_bp = Blueprint('asset_training', __name__, url_prefix='/asset-training')

@asset_training_bp.route('/simple')
@login_required
def simple():
    """Simple asset training page without complex template"""
    try:
        trainer = AssetFaceTrainer()
        
        # Get asset statistics
        real_images_path = "assets/archive/Human Faces Dataset/Real Images"
        ai_images_path = "assets/archive/Human Faces Dataset/AI-Generated Images"
        
        real_count = 0
        ai_count = 0
        
        if os.path.exists(real_images_path):
            real_count = len([f for f in os.listdir(real_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
        if os.path.exists(ai_images_path):
            ai_count = len([f for f in os.listdir(ai_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        html = f"""
        <html>
        <head>
            <title>Simple Asset Training</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .stats {{ background: #f0f0f0; padding: 20px; margin: 20px 0; }}
                .form-group {{ margin: 15px 0; }}
                button {{ background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }}
                button:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <h1>Simple Asset Training Dashboard</h1>
            
            <div class="stats">
                <h2>Available Assets:</h2>
                <ul>
                    <li>Real Images: {real_count:,}</li>
                    <li>AI Images: {ai_count:,}</li>
                    <li>Total Images: {real_count + ai_count:,}</li>
                </ul>
            </div>
            
            <form id="trainingForm" action="/asset-training/train" method="post">
                <div class="form-group">
                    <label>Maximum Images to Process:</label>
                    <input type="number" name="max_images" value="50" min="1" max="500">
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="use_real_images" checked> Use Real Images ({real_count:,} available)
                    </label>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="use_ai_images"> Use AI Images ({ai_count:,} available)
                    </label>
                </div>
                
                <button type="submit">Start Training</button>
            </form>
            
            <div id="result" style="margin-top: 20px;"></div>
            
            <p><a href="/asset-training/debug">Debug Info</a> | <a href="/dashboard">Back to Dashboard</a></p>
            
            <script>
                document.getElementById('trainingForm').onsubmit = function(e) {{
                    e.preventDefault();
                    
                    const formData = new FormData(this);
                    const button = this.querySelector('button');
                    const result = document.getElementById('result');
                    
                    button.textContent = 'Training...';
                    button.disabled = true;
                    
                    fetch('/asset-training/train', {{
                        method: 'POST',
                        body: formData
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            result.innerHTML = '<div style="background: #d4edda; padding: 10px; border: 1px solid #c3e6cb;">' + 
                                              '<strong>Success!</strong> ' + data.message + '</div>';
                        }} else {{
                            result.innerHTML = '<div style="background: #f8d7da; padding: 10px; border: 1px solid #f5c6cb;">' + 
                                              '<strong>Error:</strong> ' + data.error + '</div>';
                        }}
                    }})
                    .catch(error => {{
                        result.innerHTML = '<div style="background: #f8d7da; padding: 10px; border: 1px solid #f5c6cb;">' + 
                                          '<strong>Error:</strong> ' + error + '</div>';
                    }})
                    .finally(() => {{
                        button.textContent = 'Start Training';
                        button.disabled = false;
                    }});
                }};
            </script>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error in simple asset training page: {e}")
        return f"Simple asset training error: {str(e)}"

@asset_training_bp.route('/debug')
@login_required
def debug():
    """Debug asset training system"""
    try:
        import os
        
        # Check asset paths
        assets_dir = "assets"
        real_images_path = "assets/archive/Human Faces Dataset/Real Images"
        ai_images_path = "assets/archive/Human Faces Dataset/AI-Generated Images"
        
        debug_info = {
            'assets_dir_exists': os.path.exists(assets_dir),
            'real_images_path_exists': os.path.exists(real_images_path),
            'ai_images_path_exists': os.path.exists(ai_images_path),
            'current_working_directory': os.getcwd(),
            'assets_dir_contents': [],
            'real_images_count': 0,
            'ai_images_count': 0
        }
        
        if os.path.exists(assets_dir):
            debug_info['assets_dir_contents'] = os.listdir(assets_dir)
            
        if os.path.exists(real_images_path):
            real_files = [f for f in os.listdir(real_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            debug_info['real_images_count'] = len(real_files)
            debug_info['real_images_sample'] = real_files[:5]
            
        if os.path.exists(ai_images_path):
            ai_files = [f for f in os.listdir(ai_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            debug_info['ai_images_count'] = len(ai_files)
            debug_info['ai_images_sample'] = ai_files[:5]
        
        # Test AssetFaceTrainer
        try:
            trainer = AssetFaceTrainer()
            debug_info['trainer_init'] = 'Success'
            debug_info['trainer_assets_available'] = trainer.check_assets_available()
        except Exception as e:
            debug_info['trainer_init'] = f'Failed: {str(e)}'
            debug_info['trainer_assets_available'] = False
        
        html = f"""
        <h1>Asset Training Debug</h1>
        <h2>Directory Check:</h2>
        <ul>
            <li>Current Directory: {debug_info['current_working_directory']}</li>
            <li>Assets Directory Exists: {debug_info['assets_dir_exists']}</li>
            <li>Real Images Path Exists: {debug_info['real_images_path_exists']}</li>
            <li>AI Images Path Exists: {debug_info['ai_images_path_exists']}</li>
        </ul>
        
        <h2>Asset Contents:</h2>
        <ul>
            <li>Assets Directory Contents: {debug_info['assets_dir_contents']}</li>
            <li>Real Images Count: {debug_info['real_images_count']}</li>
            <li>AI Images Count: {debug_info['ai_images_count']}</li>
        </ul>
        
        <h2>Trainer Status:</h2>
        <ul>
            <li>Trainer Initialization: {debug_info['trainer_init']}</li>
            <li>Assets Available: {debug_info['trainer_assets_available']}</li>
        </ul>
        
        <a href="/asset-training/">Back to Asset Training</a>
        """
        
        return html
        
    except Exception as e:
        return f"Debug failed: {str(e)}"

@asset_training_bp.route('/improved')
@login_required
def improved():
    """Improved asset training dashboard with real-time progress"""
    try:
        trainer = AssetFaceTrainer()
        
        # Get asset statistics using absolute paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        real_images_path = os.path.join(project_root, "assets", "archive", "Human Faces Dataset", "Real Images")
        ai_images_path = os.path.join(project_root, "assets", "archive", "Human Faces Dataset", "AI-Generated Images")
        
        real_count = 0
        ai_count = 0
        
        if os.path.exists(real_images_path):
            real_count = len([f for f in os.listdir(real_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
        if os.path.exists(ai_images_path):
            ai_count = len([f for f in os.listdir(ai_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        stats = {
            'real_images': real_count,
            'ai_images': ai_count,
            'total_images': real_count + ai_count,
            'assets_folder': 'assets'
        }
        
        return render_template('asset_training/dashboard_improved.html', stats=stats)
        
    except Exception as e:
        logger.error(f"Error loading improved asset training dashboard: {e}")
        flash(f"Error loading dashboard: {str(e)}", 'error')
        return redirect(url_for('index'))

@asset_training_bp.route('/')
@login_required
def index():
    """Asset training dashboard"""
    try:
        trainer = AssetFaceTrainer()
        
        # Get asset statistics using absolute paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        real_images_path = os.path.join(project_root, "assets", "archive", "Human Faces Dataset", "Real Images")
        ai_images_path = os.path.join(project_root, "assets", "archive", "Human Faces Dataset", "AI-Generated Images")
        
        real_count = 0
        ai_count = 0
        
        if os.path.exists(real_images_path):
            real_count = len([f for f in os.listdir(real_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
        if os.path.exists(ai_images_path):
            ai_count = len([f for f in os.listdir(ai_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        stats = {
            'real_images': real_count,
            'ai_images': ai_count,
            'total_images': real_count + ai_count,
            'assets_folder': 'assets'
        }
        
        return render_template('asset_training/dashboard.html', stats=stats)
        
    except Exception as e:
        logger.error(f"Error loading asset training dashboard: {e}")
        flash(f"Error loading dashboard: {str(e)}", 'error')
        return redirect(url_for('index'))

@asset_training_bp.route('/train', methods=['POST'])
@login_required
def train_model():
    """Train the face recognition model with asset images"""
    try:
        # Get training parameters
        max_images = request.form.get('max_images', type=int, default=100)
        use_real_images = request.form.get('use_real_images') == 'on'
        use_ai_images = request.form.get('use_ai_images') == 'on'
        
        if not (use_real_images or use_ai_images):
            return jsonify({
                'success': False,
                'error': 'Please select at least one image type to train with'
            })
        
        # Initialize trainer
        trainer = AssetFaceTrainer()
        
        # Start training
        result = trainer.train_from_assets(
            max_images=max_images,
            use_real_images=use_real_images,
            use_ai_images=use_ai_images
        )
        
        if result['success']:
            # Calculate additional metrics
            processed = result['processed_images']
            faces = result['faces_extracted']
            accuracy = round((faces / processed) * 100, 1) if processed > 0 else 0
            
            return jsonify({
                'success': True,
                'message': f"Training completed successfully! Processed {processed} images and detected {faces} faces with {accuracy}% detection rate.",
                'processed_images': processed,
                'faces_extracted': faces,
                'detection_accuracy': accuracy,
                'training_details': {
                    'images_per_face': round(processed / faces, 2) if faces > 0 else 0,
                    'success_rate': f"{accuracy}%",
                    'total_training_data_added': faces
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            })
            
    except Exception as e:
        logger.error(f"Error during asset training: {e}")
        return jsonify({
            'success': False,
            'error': f"Training failed: {str(e)}"
        })

@asset_training_bp.route('/status')
@login_required
def training_status():
    """Get training status and statistics"""
    try:
        # Check if assets folder exists
        assets_path = "assets/archive/Human Faces Dataset"
        
        if not os.path.exists(assets_path):
            return jsonify({
                'success': False,
                'error': 'Assets folder not found'
            })
        
        # Count available images
        real_images_path = os.path.join(assets_path, "Real Images")
        ai_images_path = os.path.join(assets_path, "AI-Generated Images")
        
        real_count = 0
        ai_count = 0
        
        if os.path.exists(real_images_path):
            real_count = len([f for f in os.listdir(real_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
        if os.path.exists(ai_images_path):
            ai_count = len([f for f in os.listdir(ai_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        return jsonify({
            'success': True,
            'assets_available': True,
            'real_images': real_count,
            'ai_images': ai_count,
            'total_images': real_count + ai_count
        })
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })
