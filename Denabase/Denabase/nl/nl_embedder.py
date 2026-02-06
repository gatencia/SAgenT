import hashlib
import json
import joblib
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

DEFAULT_NL_DIM = 128

class NLEmbedder:
    """
    Deterministic NL Embedder using TF-IDF + TruncatedSVD (LSA).
    Produces normalized embeddings.
    """
    def __init__(self, dim: int = DEFAULT_NL_DIM):
        self.dim = dim
        self.effective_dim = 0
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, 
                stop_words="english", 
                ngram_range=(1, 2), 
                min_df=1, 
                max_features=50000
            )),
            ("svd", TruncatedSVD(n_components=self.dim, random_state=0)),
            ("norm", Normalizer(norm="l2"))
        ])
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "NLEmbedder":
        """Fits the embedder on a list of texts (handles None/empty)."""
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            self.is_fitted = False
            return self

        try:
            # 1. TF-IDF to see vocabulary size
            tfidf = self.pipeline.named_steps["tfidf"]
            X_tfidf = tfidf.fit_transform(valid_texts)
            n_samples, n_features = X_tfidf.shape
            
            # 2. Determine effective_dim
            # User requirement: min(self.dim, max(2, vocab_size-1), n_samples-1)
            # Must also ensure it doesn't exceed n_features - 1 for TruncatedSVD
            eff = min(self.dim, max(2, n_features - 1), n_samples - 1)
            # Final clamp to ensure validity for sklearn
            self.effective_dim = max(1, min(eff, n_features - 1, n_samples - 1))
                
            svd = self.pipeline.named_steps["svd"]
            svd.n_components = self.effective_dim
            
            # 3. Fit the rest of the pipeline
            self.pipeline.fit(valid_texts)
            self.is_fitted = True
        except Exception as e:
            import sys
            print(f"DEBUG: NLEmbedder fit failed: {e}", file=sys.stderr)
            self.is_fitted = False
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Embeds texts. Returns (n, self.dim) array. Zeros if not fitted or empty input."""
        n = len(texts) if texts else 0
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
            
        if not self.is_fitted:
            return np.zeros((n, self.dim), dtype=np.float32)

        valid_texts = [t if t else "" for t in texts]
        
        try:
            vecs = self.pipeline.transform(valid_texts)
            # vecs has shape (n, self.effective_dim)
            
            if vecs.shape[1] < self.dim:
                # Pad with trailing zeros to reach self.dim
                pad = np.zeros((vecs.shape[0], self.dim - vecs.shape[1]), dtype=np.float32)
                vecs = np.concatenate([vecs, pad], axis=1)
                
            return vecs.astype(np.float32)
        except Exception:
            # Fallback for edge cases
            return np.zeros((n, self.dim), dtype=np.float32)

    def config_hash(self) -> str:
        """Returns a canonical hash of the configuration."""
        config = {
            "dim": self.dim,
            "pipeline": "tfidf_svd_norm",
            "tfidf": {"ngram": (1, 2), "max_features": 50000, "stop_words": "english"},
            "svd": {"random_state": 0}
        }
        s = json.dumps(config, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        """Persists the embedder."""
        state = {
            "dim": self.dim,
            "effective_dim": self.effective_dim,
            "pipeline": self.pipeline,
            "is_fitted": self.is_fitted
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: Path) -> "NLEmbedder":
        """Loads the embedder."""
        state = joblib.load(path)
        obj = cls(dim=state["dim"])
        obj.effective_dim = state.get("effective_dim", 0)
        obj.pipeline = state["pipeline"]
        obj.is_fitted = state["is_fitted"]
        return obj
