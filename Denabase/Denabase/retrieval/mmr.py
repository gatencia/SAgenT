import numpy as np
from typing import List, Dict, Any

def mmr_select(candidates: List[str], 
               base_score: Dict[str, float], 
               vecs: Dict[str, np.ndarray], 
               topk: int, 
               lamb: float = 0.7) -> List[str]:
    """
    Deterministic MMR selection.
    candidates: list of candidate IDs.
    base_score: dictionary of relevance scores (higher is better).
    vecs: dictionary of embedding vectors (normalized).
    topk: number of items to select.
    lamb: lambda parameter (0.7 means 70% relevance, 30% diversity).
    """
    if not candidates:
        return []

    # Filter candidates to those present in base_score
    pool = [c for c in candidates if c in base_score]
    # Sort pool deterministically by ID initially given same score? 
    # Actually MMR loop handles selection.
    
    selected = []
    
    while len(selected) < topk and pool:
        best_mmr = -1.0e9
        best_idx = -1
        
        # Calculate MMR for each candidate in pool
        for i, cand_id in enumerate(pool):
            rel = base_score.get(cand_id, 0.0)
            
            # Compute diversity penalty: max sim to any already selected
            max_sim = 0.0
            if selected:
                cand_vec = vecs.get(cand_id)
                # If vectors missing, assume 0 similarity (no penalty)
                if cand_vec is not None:
                     for sel_id in selected:
                         sel_vec = vecs.get(sel_id)
                         if sel_vec is not None:
                             # Cosine sim for normalized vecs is dot product
                             sim = float(np.dot(cand_vec, sel_vec))
                             if sim > max_sim:
                                 max_sim = sim
            
            mmr = lamb * rel - (1.0 - lamb) * max_sim
            
            # Update best. Break ties by ID (string comparison)
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
            elif mmr == best_mmr:
                # Deterministic tie-break: lexicographical
                if pool[i] < pool[best_idx]:
                    best_idx = i
        
        if best_idx != -1:
            selected.append(pool.pop(best_idx))
        else:
            # Should not happen if pool not empty
            break
            
    return selected
