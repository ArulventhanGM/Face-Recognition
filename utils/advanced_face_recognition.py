"""
Advanced Face Recognition Module with Custom CNN and ArcFace Loss
Implements face recognition with fallback options
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging
import math
from dataclasses import dataclass
import os

# Try to import advanced libraries, fallback if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using basic face recognition")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("face_recognition library not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecognitionResult:
    """Face recognition result with embedding and confidence"""
    embedding: np.ndarray
    confidence: float
    identity: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None

class ArcFaceLoss(nn.Module):
    """ArcFace loss implementation for face recognition"""
    
    def __init__(self, embedding_size=512, num_classes=1000, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        """Forward pass of ArcFace loss"""
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # Calculate angle
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CustomFaceNet(nn.Module):
    """Custom CNN for face recognition with ArcFace training"""
    
    def __init__(self, embedding_size=512, num_classes=1000):
        super(CustomFaceNet, self).__init__()
        self.embedding_size = embedding_size
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # ArcFace loss
        self.arcface = ArcFaceLoss(embedding_size, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with residual blocks"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, labels=None):
        """Forward pass"""
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Generate embeddings
        embeddings = self.embedding(x)
        
        # During training, apply ArcFace loss
        if self.training and labels is not None:
            output = self.arcface(embeddings, labels)
            return embeddings, output
        
        return embeddings

class AdvancedFaceRecognizer:
    """Advanced face recognition system with fallback options"""
    
    def __init__(self, embedding_size=512, device='cpu'):
        self.embedding_size = embedding_size
        self.device = device
        self.model = None
        self.known_embeddings = {}
        self.known_names = []
        self.embedding_matrix = None
        self.use_torch = TORCH_AVAILABLE
        self.use_face_recognition = FACE_RECOGNITION_AVAILABLE
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the face recognition model with fallbacks"""
        try:
            if self.use_torch:
                self.model = CustomFaceNet(embedding_size=self.embedding_size)
                
                # Try to load pre-trained weights
                model_path = 'models/custom_facenet_arcface.pth'
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Custom FaceNet model loaded successfully")
                else:
                    logger.warning("Pre-trained model not found, using random weights")
                
                self.model.to(self.device)
                self.model.eval()
                logger.info("PyTorch-based recognition initialized")
                
            elif self.use_face_recognition:
                logger.info("Using face_recognition library as fallback")
            else:
                logger.warning("No recognition backends available. Using basic feature extraction.")
                
        except Exception as e:
            logger.error(f"Error initializing recognition model: {e}")
            # Fallback to basic mode
            self.use_torch = False
            if FACE_RECOGNITION_AVAILABLE:
                self.use_face_recognition = True
                logger.info("Falling back to face_recognition library")
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using available method"""
        try:
            if self.use_torch and self.model is not None:
                return self._extract_embedding_torch(face_image)
            elif self.use_face_recognition:
                return self._extract_embedding_face_recognition(face_image)
            else:
                return self._extract_embedding_basic(face_image)
                
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def _extract_embedding_torch(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using PyTorch model"""
        try:
            face_tensor = self.preprocess_face(face_image)
            if face_tensor is None:
                return None
            
            with torch.no_grad():
                embedding = self.model(face_tensor)
                
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error in PyTorch embedding extraction: {e}")
            return None
    
    def _extract_embedding_face_recognition(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using face_recognition library"""
        try:
            # Convert BGR to RGB if needed
            if len(face_image.shape) == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Extract face encodings
            encodings = face_recognition.face_encodings(rgb_image)
            if encodings:
                return encodings[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in face_recognition embedding extraction: {e}")
            return None
    
    def _extract_embedding_basic(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract basic features as fallback"""
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Resize to fixed size
            resized = cv2.resize(gray, (64, 64))
            
            # Extract basic features
            # Histogram of gradients
            sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
            
            # Create feature vector
            features = []
            features.extend(resized.flatten()[:256])  # Pixel intensities (reduced)
            features.extend(np.histogram(sobelx, bins=32)[0])  # Gradient histogram X
            features.extend(np.histogram(sobely, bins=32)[0])  # Gradient histogram Y
            
            # Normalize
            embedding = np.array(features, dtype=np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Pad or truncate to desired size
            if len(embedding) > self.embedding_size:
                embedding = embedding[:self.embedding_size]
            else:
                padding = np.zeros(self.embedding_size - len(embedding))
                embedding = np.concatenate([embedding, padding])
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error in basic embedding extraction: {e}")
            return None
    
    def preprocess_face(self, face_image: np.ndarray, target_size=(112, 112)) -> torch.Tensor:
        """Preprocess face image for recognition"""
        try:
            # Resize
            face_resized = cv2.resize(face_image, target_size)
            
            # Normalize to [-1, 1]
            face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
            
            # Convert to tensor
            if len(face_normalized.shape) == 3:
                face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1))
            else:
                face_tensor = torch.from_numpy(np.expand_dims(face_normalized, 0))
            
            # Add batch dimension
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            return face_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return None
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using the custom model"""
        try:
            if self.model is None:
                return None
            
            # Preprocess
            face_tensor = self.preprocess_face(face_image)
            if face_tensor is None:
                return None
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                
            # Normalize embedding
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def add_known_face(self, face_image: np.ndarray, name: str) -> bool:
        """Add a known face to the database"""
        try:
            embedding = self.extract_embedding(face_image)
            if embedding is None:
                return False
            
            # Store embedding
            if name not in self.known_embeddings:
                self.known_embeddings[name] = []
            
            self.known_embeddings[name].append(embedding)
            
            if name not in self.known_names:
                self.known_names.append(name)
            
            # Update embedding matrix
            self._update_embedding_matrix()
            
            logger.info(f"Added face for {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding known face: {e}")
            return False
    
    def _update_embedding_matrix(self):
        """Update the embedding matrix for efficient matching"""
        try:
            all_embeddings = []
            self.embedding_labels = []
            
            for name, embeddings in self.known_embeddings.items():
                for embedding in embeddings:
                    all_embeddings.append(embedding)
                    self.embedding_labels.append(name)
            
            if all_embeddings:
                self.embedding_matrix = np.vstack(all_embeddings)
            else:
                self.embedding_matrix = None
                
        except Exception as e:
            logger.error(f"Error updating embedding matrix: {e}")
    
    def recognize_face(self, face_image: np.ndarray, threshold: float = 0.6) -> RecognitionResult:
        """Recognize a face using cosine similarity matching"""
        try:
            # Extract embedding
            query_embedding = self.extract_embedding(face_image)
            if query_embedding is None:
                return RecognitionResult(
                    embedding=np.array([]),
                    confidence=0.0,
                    identity="Unknown"
                )
            
            if self.embedding_matrix is None or len(self.embedding_labels) == 0:
                return RecognitionResult(
                    embedding=query_embedding,
                    confidence=0.0,
                    identity="Unknown"
                )
            
            # Calculate cosine similarities
            similarities = self._cosine_similarity_vectorized(
                query_embedding.reshape(1, -1), 
                self.embedding_matrix
            ).flatten()
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= threshold:
                identity = self.embedding_labels[best_idx]
                confidence = best_similarity
            else:
                identity = "Unknown"
                confidence = best_similarity
            
            return RecognitionResult(
                embedding=query_embedding,
                confidence=float(confidence),
                identity=identity
            )
            
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return RecognitionResult(
                embedding=np.array([]),
                confidence=0.0,
                identity="Error"
            )
    
    def _cosine_similarity_vectorized(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity calculation"""
        # Normalize vectors
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        return np.dot(a_norm, b_norm.T)
    
    def train_model(self, training_data: List[Tuple[np.ndarray, str]], 
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the model with ArcFace loss"""
        try:
            if self.model is None:
                logger.error("Model not initialized")
                return
            
            # Prepare data
            images, labels = zip(*training_data)
            unique_labels = list(set(labels))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            # Update model for correct number of classes
            self.model.arcface.num_classes = len(unique_labels)
            self.model.arcface.weight = nn.Parameter(
                torch.FloatTensor(len(unique_labels), self.embedding_size)
            )
            nn.init.xavier_uniform_(self.model.arcface.weight)
            
            # Setup training
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            logger.info(f"Starting training for {epochs} epochs...")
            
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                # Process in batches
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]
                    
                    # Preprocess batch
                    batch_tensors = []
                    batch_label_indices = []
                    
                    for img, label in zip(batch_images, batch_labels):
                        tensor = self.preprocess_face(img)
                        if tensor is not None:
                            batch_tensors.append(tensor)
                            batch_label_indices.append(label_to_idx[label])
                    
                    if not batch_tensors:
                        continue
                    
                    # Convert to batch tensor
                    batch_tensor = torch.cat(batch_tensors, dim=0)
                    label_tensor = torch.tensor(batch_label_indices, device=self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    embeddings, output = self.model(batch_tensor, label_tensor)
                    
                    # Calculate loss
                    loss = criterion(output, label_tensor)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Switch back to evaluation mode
            self.model.eval()
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
    
    def save_model(self, path: str):
        """Save the trained model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'embedding_size': self.embedding_size,
                'known_embeddings': self.known_embeddings,
                'known_names': self.known_names
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.known_embeddings = checkpoint.get('known_embeddings', {})
            self.known_names = checkpoint.get('known_names', [])
            self._update_embedding_matrix()
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

# Factory function for easy initialization
def create_face_recognizer(embedding_size=512, device='cpu') -> AdvancedFaceRecognizer:
    """Create and return an initialized face recognizer"""
    return AdvancedFaceRecognizer(embedding_size=embedding_size, device=device)
