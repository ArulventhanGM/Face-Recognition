#!/usr/bin/env python3
"""
Asset Face Training System
Processes images from assets folder for face recognition training
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetFaceTrainer:
    """Simple asset-based face training system"""
    
    def __init__(self):
        # Use absolute paths to avoid path resolution issues
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assets_dir = os.path.join(project_root, "assets")
        self.real_images_dir = os.path.join(
            self.assets_dir, 
            "archive", 
            "Human Faces Dataset", 
            "Real Images"
        )
        
        logger.info(f"AssetFaceTrainer initialized:")
        logger.info(f"  Project root: {project_root}")
        logger.info(f"  Assets dir: {self.assets_dir}")
        logger.info(f"  Real images dir: {self.real_images_dir}")
        logger.info(f"  Assets dir exists: {os.path.exists(self.assets_dir)}")
        logger.info(f"  Real images dir exists: {os.path.exists(self.real_images_dir)}")
        
    def train_from_assets(self, max_images=100, use_real_images=True, use_ai_images=False):
        """Train the face recognition model using asset images"""
        try:
            # Check if assets are available
            if not self.check_assets_available():
                return {
                    'success': False,
                    'error': 'Assets directory not found or empty',
                    'processed_images': 0,
                    'faces_extracted': 0
                }
            
            processed_images = 0
            faces_extracted = 0
            
            # Process real images if requested
            if use_real_images and os.path.exists(self.real_images_dir):
                real_files = [f for f in os.listdir(self.real_images_dir) if f.lower().endswith('.jpg')]
                real_sample = real_files[:max_images // (1 if not use_ai_images else 2)]
                
                for filename in tqdm(real_sample, desc="Processing real images"):
                    try:
                        image_path = os.path.join(self.real_images_dir, filename)
                        img = cv2.imread(image_path)
                        
                        if img is not None:
                            processed_images += 1
                            # Simple face detection (this is a placeholder)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            faces_extracted += len(faces)
                            
                    except Exception as e:
                        logger.warning(f"Error processing {filename}: {e}")
                        continue
            
            # Process AI images if requested (and if the directory exists)
            ai_images_dir = os.path.join(
                self.assets_dir, 
                "archive", 
                "Human Faces Dataset", 
                "AI-Generated Images"
            )
            
            if use_ai_images and os.path.exists(ai_images_dir):
                ai_files = [f for f in os.listdir(ai_images_dir) if f.lower().endswith('.jpg')]
                ai_sample = ai_files[:max_images // (1 if not use_real_images else 2)]
                
                for filename in tqdm(ai_sample, desc="Processing AI images"):
                    try:
                        image_path = os.path.join(ai_images_dir, filename)
                        img = cv2.imread(image_path)
                        
                        if img is not None:
                            processed_images += 1
                            # Simple face detection (this is a placeholder)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            faces_extracted += len(faces)
                            
                    except Exception as e:
                        logger.warning(f"Error processing {filename}: {e}")
                        continue
            
            return {
                'success': True,
                'processed_images': processed_images,
                'faces_extracted': faces_extracted,
                'message': f'Successfully processed {processed_images} images and extracted {faces_extracted} faces'
            }
            
        except Exception as e:
            logger.error(f"Error during asset training: {e}")
            return {
                'success': False,
                'error': str(e),
                'processed_images': 0,
                'faces_extracted': 0
            }

    def check_assets_available(self):
        """Check if assets directory and images are available"""
        if not os.path.exists(self.assets_dir):
            logger.warning("Assets directory not found")
            return False
            
        if not os.path.exists(self.real_images_dir):
            logger.warning("Real images directory not found")
            return False
            
        image_files = [f for f in os.listdir(self.real_images_dir) if f.lower().endswith('.jpg')]
        logger.info(f"Found {len(image_files)} images in assets directory")
        return len(image_files) > 0
    
    def process_sample_images(self, max_images=100):
        """Process a sample of images for demonstration"""
        if not self.check_assets_available():
            return {"success": False, "message": "Assets not available"}
        
        try:
            image_files = [f for f in os.listdir(self.real_images_dir) if f.lower().endswith('.jpg')]
            sample_files = image_files[:max_images]  # Process first N images
            
            processed_count = 0
            valid_faces = 0
            
            for filename in tqdm(sample_files, desc="Processing images"):
                try:
                    image_path = os.path.join(self.real_images_dir, filename)
                    img = cv2.imread(image_path)
                    
                    if img is not None:
                        # Simple face detection using OpenCV
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                        
                        if len(faces) > 0:
                            valid_faces += 1
                        
                        processed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    continue
            
            return {
                "success": True,
                "processed": processed_count,
                "valid_faces": valid_faces,
                "message": f"Processed {processed_count} images, found {valid_faces} valid faces"
            }
            
        except Exception as e:
            logger.error(f"Error in asset training: {e}")
            return {"success": False, "message": str(e)}

def get_asset_trainer():
    """Get asset trainer instance"""
    return AssetFaceTrainer()

# For backward compatibility
def get_asset_training_system():
    """Get asset training system (backward compatibility)"""
    return get_asset_trainer()

if __name__ == "__main__":
    trainer = get_asset_trainer()
    result = trainer.process_sample_images(50)
    print(f"Training result: {result}")
