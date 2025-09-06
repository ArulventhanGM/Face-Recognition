"""
Advanced Face Matching System with Cosine Similarity and SVM
Implements sophisticated matching algorithms for face recognition
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Optional, Dict, Union
import logging
import pickle
import os
from dataclasses import dataclass
from scipy.spatial.distance import euclidean, manhattan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Face matching result with detailed metrics"""
    identity: str
    confidence: float
    similarity_score: float
    distance: float
    method_used: str
    all_scores: Optional[Dict[str, float]] = None

class AdvancedFaceMatcher:
    """Advanced face matching system with multiple algorithms"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.svm_classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.known_embeddings = {}
        self.embedding_matrix = None
        self.labels = []
        self.is_trained = False
        
    def add_face_embedding(self, embedding: np.ndarray, identity: str):
        """Add a face embedding to the database"""
        try:
            if identity not in self.known_embeddings:
                self.known_embeddings[identity] = []
            
            # Normalize embedding
            normalized_embedding = self._normalize_embedding(embedding)
            self.known_embeddings[identity].append(normalized_embedding)
            
            # Update matrices
            self._update_matrices()
            
            logger.info(f"Added embedding for {identity}")
            
        except Exception as e:
            logger.error(f"Error adding face embedding: {e}")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _update_matrices(self):
        """Update embedding matrix and labels for efficient computation"""
        try:
            all_embeddings = []
            all_labels = []
            
            for identity, embeddings in self.known_embeddings.items():
                for embedding in embeddings:
                    all_embeddings.append(embedding)
                    all_labels.append(identity)
            
            if all_embeddings:
                self.embedding_matrix = np.vstack(all_embeddings)
                self.labels = all_labels
            else:
                self.embedding_matrix = None
                self.labels = []
                
        except Exception as e:
            logger.error(f"Error updating matrices: {e}")
    
    def train_svm_classifier(self, cv_folds: int = 5, 
                           param_grid: Optional[Dict] = None) -> float:
        """Train SVM classifier with hyperparameter tuning"""
        try:
            if self.embedding_matrix is None or len(self.labels) == 0:
                logger.error("No embeddings available for training")
                return 0.0
            
            # Default parameter grid for SVM
            if param_grid is None:
                param_grid = {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
                    'classifier__kernel': ['rbf', 'linear', 'poly']
                }
            
            # Create pipeline with scaling and SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, random_state=42))
            ])
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(self.labels)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=cv_folds, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            logger.info("Training SVM classifier with grid search...")
            grid_search.fit(self.embedding_matrix, encoded_labels)
            
            # Store best model
            self.svm_classifier = grid_search.best_estimator_
            self.scaler = self.svm_classifier.named_steps['scaler']
            
            # Calculate cross-validation accuracy
            cv_scores = cross_val_score(
                self.svm_classifier, 
                self.embedding_matrix, 
                encoded_labels, 
                cv=cv_folds
            )
            
            accuracy = np.mean(cv_scores)
            self.is_trained = True
            
            logger.info(f"SVM training completed. Best params: {grid_search.best_params_}")
            logger.info(f"Cross-validation accuracy: {accuracy:.4f} (+/- {np.std(cv_scores) * 2:.4f})")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training SVM classifier: {e}")
            return 0.0
    
    def match_cosine_similarity(self, query_embedding: np.ndarray, 
                              threshold: float = 0.7) -> MatchResult:
        """Match face using cosine similarity"""
        try:
            if self.embedding_matrix is None:
                return MatchResult(
                    identity="Unknown",
                    confidence=0.0,
                    similarity_score=0.0,
                    distance=float('inf'),
                    method_used="cosine_similarity"
                )
            
            # Normalize query embedding
            query_normalized = self._normalize_embedding(query_embedding)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(
                query_normalized.reshape(1, -1), 
                self.embedding_matrix
            ).flatten()
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            best_identity = self.labels[best_idx]
            
            # Calculate confidence based on threshold
            confidence = min(best_similarity / threshold, 1.0) if threshold > 0 else best_similarity
            
            # Determine final identity
            final_identity = best_identity if best_similarity >= threshold else "Unknown"
            
            return MatchResult(
                identity=final_identity,
                confidence=float(confidence),
                similarity_score=float(best_similarity),
                distance=1.0 - best_similarity,
                method_used="cosine_similarity"
            )
            
        except Exception as e:
            logger.error(f"Error in cosine similarity matching: {e}")
            return MatchResult(
                identity="Error",
                confidence=0.0,
                similarity_score=0.0,
                distance=float('inf'),
                method_used="cosine_similarity"
            )
    
    def match_euclidean_distance(self, query_embedding: np.ndarray, 
                               threshold: float = 0.8) -> MatchResult:
        """Match face using Euclidean distance"""
        try:
            if self.embedding_matrix is None:
                return MatchResult(
                    identity="Unknown",
                    confidence=0.0,
                    similarity_score=0.0,
                    distance=float('inf'),
                    method_used="euclidean_distance"
                )
            
            # Normalize query embedding
            query_normalized = self._normalize_embedding(query_embedding)
            
            # Calculate Euclidean distances
            distances = [euclidean(query_normalized, emb) for emb in self.embedding_matrix]
            
            # Find closest match
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            best_identity = self.labels[best_idx]
            
            # Convert distance to similarity
            similarity = 1.0 / (1.0 + best_distance)
            
            # Calculate confidence
            confidence = max(0.0, (threshold - best_distance) / threshold)
            
            # Determine final identity
            final_identity = best_identity if best_distance <= threshold else "Unknown"
            
            return MatchResult(
                identity=final_identity,
                confidence=float(confidence),
                similarity_score=float(similarity),
                distance=float(best_distance),
                method_used="euclidean_distance"
            )
            
        except Exception as e:
            logger.error(f"Error in Euclidean distance matching: {e}")
            return MatchResult(
                identity="Error",
                confidence=0.0,
                similarity_score=0.0,
                distance=float('inf'),
                method_used="euclidean_distance"
            )
    
    def match_svm_classifier(self, query_embedding: np.ndarray, 
                           threshold: float = 0.7) -> MatchResult:
        """Match face using trained SVM classifier"""
        try:
            if not self.is_trained or self.svm_classifier is None:
                logger.warning("SVM classifier not trained. Training now...")
                accuracy = self.train_svm_classifier()
                if accuracy == 0.0:
                    return MatchResult(
                        identity="Unknown",
                        confidence=0.0,
                        similarity_score=0.0,
                        distance=float('inf'),
                        method_used="svm_classifier"
                    )
            
            # Normalize and reshape query embedding
            query_normalized = self._normalize_embedding(query_embedding).reshape(1, -1)
            
            # Predict with probabilities
            probabilities = self.svm_classifier.predict_proba(query_normalized)[0]
            predicted_class = self.svm_classifier.predict(query_normalized)[0]
            
            # Get class names
            class_names = self.label_encoder.classes_
            
            # Find best match
            best_prob = np.max(probabilities)
            best_class_idx = np.argmax(probabilities)
            best_identity = class_names[best_class_idx]
            
            # Calculate confidence
            confidence = best_prob if best_prob >= threshold else 0.0
            
            # Determine final identity
            final_identity = best_identity if best_prob >= threshold else "Unknown"
            
            return MatchResult(
                identity=final_identity,
                confidence=float(confidence),
                similarity_score=float(best_prob),
                distance=1.0 - best_prob,
                method_used="svm_classifier"
            )
            
        except Exception as e:
            logger.error(f"Error in SVM classification: {e}")
            return MatchResult(
                identity="Error",
                confidence=0.0,
                similarity_score=0.0,
                distance=float('inf'),
                method_used="svm_classifier"
            )
    
    def match_ensemble(self, query_embedding: np.ndarray, 
                      weights: Optional[Dict[str, float]] = None,
                      thresholds: Optional[Dict[str, float]] = None) -> MatchResult:
        """Match face using ensemble of methods"""
        try:
            # Default weights and thresholds
            if weights is None:
                weights = {
                    'cosine_similarity': 0.4,
                    'euclidean_distance': 0.3,
                    'svm_classifier': 0.3
                }
            
            if thresholds is None:
                thresholds = {
                    'cosine_similarity': 0.7,
                    'euclidean_distance': 0.8,
                    'svm_classifier': 0.7
                }
            
            # Get results from all methods
            cosine_result = self.match_cosine_similarity(query_embedding, thresholds['cosine_similarity'])
            euclidean_result = self.match_euclidean_distance(query_embedding, thresholds['euclidean_distance'])
            svm_result = self.match_svm_classifier(query_embedding, thresholds['svm_classifier'])
            
            # Collect all results
            results = {
                'cosine_similarity': cosine_result,
                'euclidean_distance': euclidean_result,
                'svm_classifier': svm_result
            }
            
            # Calculate weighted scores for each identity
            identity_scores = {}
            
            for method, result in results.items():
                if result.identity not in ["Unknown", "Error"]:
                    if result.identity not in identity_scores:
                        identity_scores[result.identity] = 0.0
                    
                    identity_scores[result.identity] += weights[method] * result.confidence
            
            # Find best identity
            if identity_scores:
                best_identity = max(identity_scores, key=identity_scores.get)
                best_score = identity_scores[best_identity]
                
                # Calculate overall confidence
                total_weight = sum(weights.values())
                normalized_confidence = best_score / total_weight
                
                # Collect all method scores for this identity
                all_scores = {}
                for method, result in results.items():
                    all_scores[method] = result.confidence if result.identity == best_identity else 0.0
                
                return MatchResult(
                    identity=best_identity,
                    confidence=float(normalized_confidence),
                    similarity_score=float(normalized_confidence),
                    distance=1.0 - normalized_confidence,
                    method_used="ensemble",
                    all_scores=all_scores
                )
            else:
                return MatchResult(
                    identity="Unknown",
                    confidence=0.0,
                    similarity_score=0.0,
                    distance=float('inf'),
                    method_used="ensemble"
                )
                
        except Exception as e:
            logger.error(f"Error in ensemble matching: {e}")
            return MatchResult(
                identity="Error",
                confidence=0.0,
                similarity_score=0.0,
                distance=float('inf'),
                method_used="ensemble"
            )
    
    def match_face(self, query_embedding: np.ndarray, 
                   method: str = "ensemble", **kwargs) -> MatchResult:
        """Match face using specified method"""
        method = method.lower()
        
        if method == "cosine_similarity":
            return self.match_cosine_similarity(query_embedding, **kwargs)
        elif method == "euclidean_distance":
            return self.match_euclidean_distance(query_embedding, **kwargs)
        elif method == "svm_classifier":
            return self.match_svm_classifier(query_embedding, **kwargs)
        elif method == "ensemble":
            return self.match_ensemble(query_embedding, **kwargs)
        else:
            logger.warning(f"Unknown method: {method}. Using ensemble.")
            return self.match_ensemble(query_embedding, **kwargs)
    
    def get_statistics(self) -> Dict[str, Union[int, float, List[str]]]:
        """Get statistics about the face database"""
        try:
            total_embeddings = sum(len(embeddings) for embeddings in self.known_embeddings.values())
            unique_identities = len(self.known_embeddings)
            
            # Calculate average embeddings per identity
            avg_embeddings = total_embeddings / unique_identities if unique_identities > 0 else 0
            
            # Get identity names
            identity_names = list(self.known_embeddings.keys())
            
            return {
                'total_embeddings': total_embeddings,
                'unique_identities': unique_identities,
                'average_embeddings_per_identity': avg_embeddings,
                'identity_names': identity_names,
                'is_svm_trained': self.is_trained,
                'embedding_dimension': self.embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def save_matcher(self, filepath: str):
        """Save the matcher state to file"""
        try:
            state = {
                'known_embeddings': self.known_embeddings,
                'embedding_dim': self.embedding_dim,
                'svm_classifier': self.svm_classifier,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Matcher saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving matcher: {e}")
    
    def load_matcher(self, filepath: str):
        """Load the matcher state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.known_embeddings = state['known_embeddings']
            self.embedding_dim = state['embedding_dim']
            self.svm_classifier = state['svm_classifier']
            self.scaler = state['scaler']
            self.label_encoder = state['label_encoder']
            self.is_trained = state['is_trained']
            
            # Update matrices
            self._update_matrices()
            
            logger.info(f"Matcher loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading matcher: {e}")

# Factory function for easy initialization
def create_face_matcher(embedding_dim: int = 512) -> AdvancedFaceMatcher:
    """Create and return an initialized face matcher"""
    return AdvancedFaceMatcher(embedding_dim=embedding_dim)
