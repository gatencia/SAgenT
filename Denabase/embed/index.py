import joblib
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict, Union
from sklearn.neighbors import NearestNeighbors

# Try importing FAISS
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class VectorIndex:
    """
    Vector index for similarity search.
    Uses FAISS if available, otherwise falls back to sklearn NearestNeighbors.
    """
    def __init__(self, metric: str = 'cosine'):
        self.metric = metric
        self.backend = "faiss" if HAS_FAISS else "sklearn"
        self.index = None
        self.ids: List[Any] = []
        self.vectors: Optional[np.ndarray] = None # Used for sklearn/persistence

    def add(self, vectors: List[List[float]], ids: List[Any]):
        """Adds vectors and their corresponding IDs to the index."""
        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs must have same length.")
        
        X = np.array(vectors).astype('float32')
        self.ids.extend(ids)
        
        if self.vectors is None:
            self.vectors = X
        else:
            self.vectors = np.vstack([self.vectors, X])
            
        # Rebuild index (simple approach for now)
        self._build_index()

    def _build_index(self):
        if self.vectors is None or len(self.vectors) == 0:
            return

        dim = self.vectors.shape[1]
        
        if self.backend == "faiss":
            if self.metric == 'cosine':
                # For cosine, we normalize vectors and use IP (Inner Product)
                self.index = faiss.IndexFlatIP(dim)
                # Normalize data in place (careful if persisting raw vectors?)
                # Actually, standard FAISS practice for cosine is normalize query + data
                faiss.normalize_L2(self.vectors)
                self.index.add(self.vectors)
            else:
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(self.vectors)
        else:
            # Sklearn fallback
            self.index = NearestNeighbors(n_neighbors=1, metric=self.metric, algorithm='auto')
            self.index.fit(self.vectors)

    def query(self, vector: List[float], k: int = 5) -> List[Tuple[Any, float]]:
        """
        Finds k nearest neighbors.
        Returns list of (id, similarity). 
        Similarity is 1.0 for exact matches (cosine) or distance (l2).
        For cosine, we return similarity (1 - distance) if using sklearn, or direct IP from FAISS.
        """
        if self.index is None and (self.vectors is None or len(self.vectors) == 0):
            return []
            
        k = min(k, len(self.ids))
        if k == 0:
            return []
            
        q = np.array([vector]).astype('float32')
        
        if self.backend == "faiss":
             if self.metric == 'cosine':
                 faiss.normalize_L2(q)
             
             distances, indices = self.index.search(q, k)
             # FAISS IP returns cosine similarity directly
             # FAISS L2 returns squared euclidean distance
             return [(self.ids[idx], float(dist)) for dist, idx in zip(distances[0], indices[0]) if idx != -1]
             
        else:
            # Sklearn
            distances, indices = self.index.kneighbors(q, n_neighbors=k)
            # Sklearn cosine metric returns distance = 1 - cos_sim
            # We convert back to similarity for consistency
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                sim = 1.0 - dist if self.metric == 'cosine' else dist
                results.append((self.ids[idx], float(sim)))
            return results

    def save(self, path: Union[str, Path]):
        """Persists index structure and data."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # We save the raw data and backend choice
        # We do NOT save the faiss index object directly to avoid version issues,
        # instead we rebuild on load from stored vectors.
        state = {
            "backend": self.backend,
            "metric": self.metric,
            "ids": self.ids,
            "vectors": self.vectors
        }
        joblib.dump(state, path)

    @staticmethod
    def load(path: Union[str, Path]) -> 'VectorIndex':
        if not os.path.exists(path):
             raise FileNotFoundError(f"Index file not found: {path}")
             
        state = joblib.load(path)
        idx = VectorIndex(metric=state["metric"])
        # Prefer available backend if stored one is missing? 
        # Actually logic in __init__ defaults to available. 
        # But we should respect stored preference if feasible or strict?
        # Let's use the local environment's capability primarily.
        
        idx.ids = state["ids"]
        idx.vectors = state["vectors"]
        idx._build_index()
        return idx
