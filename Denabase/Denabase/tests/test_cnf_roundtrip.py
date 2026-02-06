import os
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from pysat.solvers import Solver
from Denabase.Denabase.cnf import CnfDocument, read_dimacs, write_dimacs, simplify_cnf

# Hypothesis strategy for random CNFs
@st.composite
def cnf_strategy(draw):
    num_vars = draw(st.integers(min_value=1, max_value=50))
    num_clauses = draw(st.integers(min_value=1, max_value=100))
    
    clauses = []
    for _ in range(num_clauses):
        # random non-empty clause
        clause_len = draw(st.integers(min_value=1, max_value=5))
        clause = draw(st.lists(
            st.integers(min_value=1, max_value=num_vars).map(lambda x: x if draw(st.booleans()) else -x),
            min_size=clause_len, max_size=clause_len, unique=True
        ))
        clauses.append(clause)
    
    return CnfDocument(num_vars=num_vars, clauses=clauses)

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(doc=cnf_strategy())
def test_cnf_io_roundtrip(doc, tmp_path):
    """Verifies that writing and reading DIMACS preserves canonical form."""
    path = tmp_path / "test.cnf"
    write_dimacs(doc, path)
    
    read_doc = read_dimacs(path)
    
    assert doc.num_vars == read_doc.num_vars
    assert doc.get_canonical_clauses() == read_doc.get_canonical_clauses()

@given(doc=cnf_strategy())
def test_simplification_preserves_sat(doc):
    """Verifies that simplification preserves satisfiability using Minisat22."""
    simplified_doc, _ = simplify_cnf(doc)
    
    def is_sat(d):
        with Solver(name="minisat22") as s:
            for c in d.clauses:
                s.add_clause(c)
            return s.solve()
            
    assert is_sat(doc) == is_sat(simplified_doc)

def test_manual_tautology_removal():
    """Specific check for tautology and duplicate removal."""
    doc = CnfDocument(num_vars=3, clauses=[
        [1, -1, 2], # Tautology
        [1, 2],
        [2, 1],    # Duplicate
        [1, 1, 2]  # Duplicate literal
    ])
    simplified, report = simplify_cnf(doc)
    
    # Should only contain [1, 2] canonicalized
    assert len(simplified.clauses) == 1
    assert simplified.get_canonical_clauses() == [(1, 2)]
    assert report["removed_tautologies_and_duplicates"] == 3
