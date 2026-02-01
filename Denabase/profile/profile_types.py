import json
import hashlib
from typing import Dict, List, Any, Union
from pydantic import BaseModel, Field
from Denabase.core.serialization import to_json

class ConstraintProfile(BaseModel):
    """Deterministic structural profile of a constraint system."""
    schema_version: str = "1.0"
    counts: Dict[str, int] = Field(default_factory=dict)
    cardinalities: Dict[str, List[int]] = Field(default_factory=dict)
    graphish: Dict[str, float] = Field(default_factory=dict)
    cnf_stats: Dict[str, Any] = Field(default_factory=dict)

def profile_hash(profile: ConstraintProfile) -> str:
    """Computes a stable SHA256 hash of the profile."""
    # to_json provides robust, sorted keys serialization
    json_str = to_json(profile.model_dump())
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

def profile_jaccard(a: ConstraintProfile, b: ConstraintProfile) -> float:
    """Computes a similarity score between two profiles based on key overlap."""
    # Flatten important features for comparison
    def get_features(p: ConstraintProfile) -> Dict[str, Any]:
        feats = {}
        for k, v in p.counts.items():
            if v > 0: feats[f"count:{k}"] = v
        for k, v in p.cardinalities.items():
            if v: feats[f"card:{k}"] = tuple(sorted(v))
        return feats

    f_a = get_features(a)
    f_b = get_features(b)
    
    keys_a = set(f_a.keys())
    keys_b = set(f_b.keys())
    
    if not keys_a and not keys_b:
        return 1.0
        
    intersection = len(keys_a & keys_b)
    union = len(keys_a | keys_b)
    
    return intersection / union

def profile_vector(profile: ConstraintProfile) -> List[float]:
    """
    Returns a fixed-length numeric vector representation of the profile.
    Vector slots:
    0-9: Operator counts (And, Or, Not, Imp, Iff, Xor, Lit, AtLeast, AtMost, Exactly)
    10-12: CNF stats (vars, clauses, clauses/vars ratio)
    13-15: Cardinality stats (count, mean k, mean size)
    """
    vec = [0.0] * 16
    
    # Operator counts
    ops = ["and", "or", "not", "imp", "iff", "xor", "lit", "at_least", "at_most", "exactly"]
    for i, op in enumerate(ops):
        vec[i] = float(profile.counts.get(op, 0))
        
    # CNF stats (if present)
    if profile.cnf_stats:
        n_vars = profile.cnf_stats.get("n_vars", 0)
        n_clauses = profile.cnf_stats.get("n_clauses", 0)
        vec[10] = float(n_vars)
        vec[11] = float(n_clauses)
        vec[12] = n_clauses / n_vars if n_vars > 0 else 0.0
        
    # Cardinality summary
    k_vals = []
    sizes = []
    for k, v in profile.cardinalities.items():
        if k.endswith("_k"):
            k_vals.extend(v)
        if k.endswith("_size"):
            sizes.extend(v)
            
    vec[13] = float(len(k_vals))
    vec[14] = sum(k_vals) / len(k_vals) if k_vals else 0.0
    vec[15] = sum(sizes) / len(sizes) if sizes else 0.0
    
    return vec
