import math
import numpy as np
from typing import Dict, Any, List
from Denabase.Denabase.cnf.cnf_types import CnfDocument

def compute_gini(x: List[float]) -> float:
    """Computes the Gini coefficient of a list of values."""
    if not x:
        return 0.0
    x = sorted(x)
    n = len(x)
    if n == 0 or sum(x) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def compute_cnf_stats(doc: CnfDocument) -> Dict[str, Any]:
    """
    Computes a comprehensive set of deterministic CNF statistics.
    """
    n_vars = doc.num_vars
    n_clauses = len(doc.clauses)
    
    if n_clauses == 0:
        return {
            "n_vars": n_vars,
            "n_clauses": 0,
            "clause_len": {"min": 0, "mean": 0, "max": 0},
            "polarity_ratio": 0.5,
            "var_occurrence": {"min": 0, "mean": 0, "max": 0, "gini": 0.0},
            "clause_size_histogram": {i: 0 for i in range(1, 11)} | {"overflow": 0}
        }

    clause_lens = [len(c) for c in doc.clauses]
    
    # Polarities
    pos_lits = 0
    total_lits = sum(clause_lens)
    var_counts = {}
    
    for clause in doc.clauses:
        for lit in clause:
            if lit > 0:
                pos_lits += 1
            v = abs(lit)
            var_counts[v] = var_counts.get(v, 0) + 1
            
    # Polarity ratio: fraction of positive literals
    polarity_ratio = pos_lits / total_lits if total_lits > 0 else 0.5
    
    # Occurrence stats
    occurrences = list(var_counts.values())
    # Fill in variables with 0 occurrences
    n_active_vars = len(var_counts)
    if n_vars > n_active_vars:
        occurrences.extend([0] * (n_vars - n_active_vars))
        
    # Histogram
    hist = {i: 0 for i in range(1, 11)}
    overflow = 0
    for length in clause_lens:
        if length <= 10:
            hist[length] += 1
        else:
            overflow += 1
    hist["overflow"] = overflow

    return {
        "n_vars": n_vars,
        "n_clauses": n_clauses,
        "clause_len": {
            "min": min(clause_lens),
            "mean": float(np.mean(clause_lens)),
            "max": max(clause_lens)
        },
        "polarity_ratio": polarity_ratio,
        "var_occurrence": {
            "min": min(occurrences) if occurrences else 0,
            "mean": float(np.mean(occurrences)) if occurrences else 0,
            "max": max(occurrences) if occurrences else 0,
            "gini": float(compute_gini(occurrences))
        },
        "clause_size_histogram": hist
    }
