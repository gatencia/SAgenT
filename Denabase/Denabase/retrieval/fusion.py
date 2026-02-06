from typing import Dict

def clamp_alpha(x: float) -> float:
    """Clamps alpha to [0, 1]."""
    return max(0.0, min(1.0, float(x)))

def fuse_scores(struct_scores: Dict[str, float], 
               nl_scores: Dict[str, float], 
               alpha: float) -> Dict[str, float]:
    """
    Fuses structural and NL scores.
    Final = alpha * struct + (1 - alpha) * nl
    Missing scores treated as 0.0.
    """
    alpha = clamp_alpha(alpha)
    all_ids = set(struct_scores.keys()) | set(nl_scores.keys())
    fused = {}
    
    for eid in all_ids:
        s = struct_scores.get(eid, 0.0)
        n = nl_scores.get(eid, 0.0)
        fused[eid] = alpha * s + (1.0 - alpha) * n
        
    return fused
