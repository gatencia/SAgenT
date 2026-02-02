from typing import Any, List, Dict
from Denabase.cnf.cnf_types import CnfDocument
from Denabase.cnf.cnf_stats import compute_cnf_stats
from Denabase.profile.profile_types import ConstraintProfile
from Denabase.ir.ir_types import IR, Exactly, AtMost, AtLeast, And, Or, Not, Xor, Iff

def compute_cnf_profile(doc: CnfDocument) -> ConstraintProfile:
    """Computes a profile from a CNF document."""
    stats = compute_cnf_stats(doc)
    
    # Counts can be partially derived or left empty if strictly CNF
    counts = {
        "clauses": len(doc.clauses),
        "vars": doc.num_vars
    }
    
    # Approximate graphish metrics
    graphish = {
        "density": stats["n_clauses"] / stats["n_vars"] if stats["n_vars"] > 0 else 0.0,
        "gini_occ": stats["var_occurrence"]["gini"]
    }
    
    return ConstraintProfile(
        counts=counts,
        cnf_stats=stats,
        graphish=graphish
    )

from Denabase.profile.ir_profile import compute_ir_profile as _compute_ir_profile

def compute_ir_profile(ir_obj: Any) -> ConstraintProfile:
    """Computes a profile from an IR object (high-level stats)."""
    return _compute_ir_profile(ir_obj)
