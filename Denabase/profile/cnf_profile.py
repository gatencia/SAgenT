from Denabase.cnf.cnf_types import CnfDocument
from Denabase.cnf.cnf_stats import compute_cnf_stats
from Denabase.profile.profile_types import ConstraintProfile

def compute_cnf_profile(doc: CnfDocument) -> ConstraintProfile:
    """Computes a profile from a CNF document."""
    stats = compute_cnf_stats(doc)
    
    # Counts can be partially derived or left empty if strictly CNF
    # For CNF, "and" is implicit between clauses, "or" within clauses
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
