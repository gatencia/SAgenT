import joblib
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from Denabase.Denabase.core.errors import EmbeddingError
from Denabase.Denabase.cnf.cnf_types import CnfDocument
from Denabase.Denabase.profile.profile_types import ConstraintProfile
from Denabase.Denabase.graph.fingerprints import compute_fingerprint

class FeatureEmbedder:
    """
    Pipelines feature scaling and dimensionality reduction.
    If not fitted, falls back to a deterministic random projection.
    """
    def __init__(self, output_dim: int = 64, random_state: int = 42):
        self.output_dim = output_dim
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=output_dim, random_state=random_state)
        self.fitted = False
        
        # Fallback projection matrix (lazy init)
        self._projection_matrix: Optional[np.ndarray] = None
        self._fallback_rng = np.random.RandomState(random_state)

    def _get_raw_vector(self, doc: CnfDocument, profile: Optional[ConstraintProfile] = None) -> List[float]:
        """Extracts the raw feature vector from a CnfDocument."""
        try:
            fp = compute_fingerprint(doc, profile)
            return fp.feature_vector
        except Exception as e:
            raise EmbeddingError(f"Failed to compute fingerprint for document: {e}")

    def _deterministic_project(self, vectors: List[List[float]]) -> np.ndarray:
        """
        Projects vectors using a deterministic random matrix.
        Output is L2 normalized.
        """
        X = np.array(vectors, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        n_features = X.shape[1]
        
        # Initialize projection matrix if needed or if dimensions changed
        # Note: sparse random projection is efficient, here we do simple dense normal for robustness/simplicity
        # as requested "dense normal matrix... must be deterministic"
        if self._projection_matrix is None or self._projection_matrix.shape[0] != n_features:
            # Re-seed to ensure same matrix for same input dims
            rng = np.random.RandomState(self.random_state)
            # Create (n_features, output_dim) matrix
            self._projection_matrix = rng.normal(loc=0.0, scale=1.0/np.sqrt(self.output_dim), size=(n_features, self.output_dim)).astype(np.float32)
            
        projected = X @ self._projection_matrix
        
        # L2 normalize
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        # Avoid div by zero
        norms[norms == 0] = 1e-10
        projected = projected / norms
        
        return projected

    def fit(self, vectors: List[List[float]]) -> 'FeatureEmbedder':
        """Fits the scaler and PCA to the provided vectors."""
        if not vectors:
            raise EmbeddingError("Cannot fit on empty vector list.")
            
        X = np.array(vectors)
        if X.ndim != 2:
             raise EmbeddingError(f"Input vectors must be 2D, got shape {X.shape}")
             
        try:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Adjust n_components if necessary
            n_samples, n_features = X_scaled.shape
            n_components = min(self.output_dim, n_features, n_samples)
            
            # Create fresh PCA with correct components
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            
            self.pca.fit(X_scaled)
            self.fitted = True
        except Exception as e:
            raise EmbeddingError(f"Fitting failed: {e}")
            
        return self

    def transform(self, vectors: List[List[float]]) -> List[List[float]]:
        """Transforms vectors using the fitted pipeline."""
        if not self.fitted:
            raise EmbeddingError("FeatureEmbedder must be fitted before transform.")
            
        try:
            X = np.array(vectors)
            if X.ndim != 2:
                 if X.ndim == 1:
                     X = X.reshape(1, -1)
            
            X_scaled = self.scaler.transform(X)
            X_reduced = self.pca.transform(X_scaled)
            return X_reduced.tolist()
        except Exception as e:
             raise EmbeddingError(f"Transform failed: {e}")

    def fit_transform(self, vectors: List[List[float]]) -> List[List[float]]:
        """Fits and transforms in one step."""
        self.fit(vectors)
        return self.transform(vectors)

    def embed(self, doc: CnfDocument, profile: Optional[ConstraintProfile] = None) -> np.ndarray:
        """
        Embeds a single CnfDocument into a fixed-length vector.
        Returns a 1D float32 numpy array.
        """
        if not isinstance(doc, CnfDocument):
             raise EmbeddingError(f"Expected CnfDocument, got {type(doc)}")
             
        raw_vec = self._get_raw_vector(doc, profile)
        
        if not self.fitted:
             # Use deterministic fallback
             projected = self._deterministic_project([raw_vec])
             return projected[0]
        
        # transform expects 2D
        transformed = self.transform([raw_vec])
        return np.array(transformed[0], dtype=np.float32)

    def batch_embed(self, docs: List[CnfDocument], profiles: Optional[List[ConstraintProfile]] = None) -> np.ndarray:
        """
        Embeds a list of CnfDocuments.
        Returns a 2D float32 numpy array (n_samples, output_dim).
        """
        if not docs:
            return np.array([], dtype=np.float32)
            
        if profiles and len(profiles) != len(docs):
            raise EmbeddingError("Length of profiles must match length of docs.")
            
        raw_vecs = []
        for i, doc in enumerate(docs):
            prof = profiles[i] if profiles else None
            # Validation done inside _get_raw_vector caller logic usually, but here:
            if not isinstance(doc, CnfDocument):
                 raise EmbeddingError(f"Item at index {i} is not a CnfDocument")
            raw_vecs.append(self._get_raw_vector(doc, prof))
            
        if not self.fitted:
             return self._deterministic_project(raw_vecs)
             
        transformed = self.transform(raw_vecs)
        return np.array(transformed, dtype=np.float32)

    def save(self, path: Union[str, Path]):
        """Persists the embedder to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Union[str, Path]) -> 'FeatureEmbedder':
        """Loads the embedder from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedder file not found: {path}")
        return joblib.load(path)
