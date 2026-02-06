import hashlib
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union
from pydantic import BaseModel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Denabase.Denabase.profile.profile_types import ConstraintProfile

class AlphaExample(BaseModel):
    features: Dict[str, Union[float, int, str]]
    alpha: float

def extract_alpha_features(profile: ConstraintProfile, *, family: str = None, has_nl: bool = False) -> Dict[str, Union[float, int, str]]:
    """
    Canonical flattening of a ConstraintProfile into scalar features for AlphaModel.
    """
    stats = profile.cnf_stats or {}
    
    feats = {}
    feats["n_vars"] = int(stats.get("n_vars", 0))
    feats["n_clauses"] = int(stats.get("n_clauses", 0))
    
    # Nested extraction for mean_clause_len
    clause_len = stats.get("clause_len", {})
    if isinstance(clause_len, dict):
        feats["mean_clause_len"] = float(clause_len.get("mean", 0.0))
    else:
        feats["mean_clause_len"] = 0.0
        
    feats["polarity_ratio"] = float(stats.get("polarity_ratio", 0.5))
    feats["has_nl"] = 1 if has_nl else 0
    feats["family"] = str(family or "unknown")
    
    return feats

class AlphaModel:
    """
    Lightweight model to predict alpha (structural trust) given query features.
    Uses Ridge regression.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42)) 
        ])
        self.is_fitted = False
        self.feature_names = [] # To ensure stable ordering

    def _extract_vector(self, feats: Dict[str, Any]) -> np.ndarray:
        """Converts feature dict to fixed vector."""
        # Simple extraction logic
        # 1. n_vars
        # 2. n_clauses
        # 3. clause_len_mean
        # 4. polarity_ratio
        # 5. has_nl (0/1)
        # 6. family hashing (32 dims)
        
        vec = []
        vec.append(float(feats.get("n_vars", 0)))
        vec.append(float(feats.get("n_clauses", 0)))
        
        # Handle summary stats nesting if passed, or direct keys
        # Caller should flatten
        vec.append(float(feats.get("mean_clause_len", 0.0)))
        vec.append(float(feats.get("polarity_ratio", 0.5)))
        vec.append(float(feats.get("has_nl", 0)))
        
        # Family hashing
        fam = str(feats.get("family", "unknown"))
        h = int(hashlib.md5(fam.encode("utf-8")).hexdigest(), 16)
        bucket = h % 32
        fam_vec = [0.0] * 32
        fam_vec[bucket] = 1.0
        vec.extend(fam_vec)
        
        return np.array(vec, dtype=np.float32)

    def fit(self, examples: List[AlphaExample]) -> "AlphaModel":
        if not examples:
            return self

        X = np.stack([self._extract_vector(ex.features) for ex in examples])
        y = np.array([ex.alpha for ex in examples])
        
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_alpha(self, features: Dict[str, Any]) -> float:
        """Predicts alpha in [0, 1]."""
        if not self.is_fitted:
            return 0.7 # Default fallback
            
        x = self._extract_vector(features).reshape(1, -1)
        pred = self.pipeline.predict(x)[0]
        return max(0.0, min(1.0, float(pred)))

    def model_hash(self) -> str:
        s = "AlphaModel:Ridge:v1"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "AlphaModel":
        return joblib.load(path)
