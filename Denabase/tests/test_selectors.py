from Denabase.selection.encoding_selector import EncodingSelector
from Denabase.selection.solver_selector import SolverSelector
from Denabase.profile.profile_types import ConstraintProfile

def test_encoding_selection_heuristics():
    selector = EncodingSelector()
    
    # Case 1: Large cardinality group
    p_large = ConstraintProfile(
        cardinalities={"all_size": [100], "all_k": [5]}
    )
    r_large = selector.select(p_large)
    assert r_large.cardinality_encoding == "sequential"
    assert "sequential" in r_large.notes[0]
    
    # Case 2: Small cardinality group
    p_small = ConstraintProfile(
        cardinalities={"all_size": [5], "all_k": [1]}
    )
    r_small = selector.select(p_small)
    assert r_small.cardinality_encoding == "pairwise"
    
    # Case 3: High density
    p_dense = ConstraintProfile(
        graphish={"density": 0.8}
    )
    r_dense = selector.select(p_dense)
    assert r_dense.symmetry_breaking is True

def test_solver_selection_heuristics():
    selector = SolverSelector()
    
    # Case 1: Large Instance
    p_large = ConstraintProfile(
        cnf_stats={"n_vars": 20000, "n_clauses": 100000}
    )
    s_large = selector.select(p_large)
    # Expect Cadical first
    assert len(s_large) >= 1
    assert s_large[0].solver_name == "cadical153"
    
    # Case 2: Small Instance
    p_small = ConstraintProfile(
        cnf_stats={"n_vars": 100, "n_clauses": 400}
    )
    s_small = selector.select(p_small)
    # Expect Glucose or Minisat
    assert s_small[0].solver_name in ["glucose3", "m22"]
