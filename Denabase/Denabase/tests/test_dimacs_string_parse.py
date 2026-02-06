import pytest
from Denabase.Denabase.cnf.cnf_io import read_dimacs_from_string
from Denabase.Denabase.core.errors import CNFError

def test_dimacs_simple_sat():
    """Case 1: simple SAT CNF in one line."""
    dimacs = "p cnf 2 1\n1 2 0"
    doc = read_dimacs_from_string(dimacs)
    assert doc.num_vars == 2
    assert doc.clauses == [[1, 2]]

def test_dimacs_simple_unsat():
    """Case 2: UNSAT CNF with two unit clauses."""
    dimacs = """
    c Simple UNSAT
    p cnf 1 2
    1 0
    -1 0
    """
    doc = read_dimacs_from_string(dimacs)
    assert doc.num_vars == 1
    assert doc.clauses == [[1], [-1]]

def test_dimacs_clause_spanning_lines():
    """Case 3: clause spanning lines."""
    dimacs = """
    p cnf 3 1
    1 -2
    3 0
    """
    doc = read_dimacs_from_string(dimacs)
    assert doc.num_vars == 3
    assert doc.clauses == [[1, -2, 3]]

def test_dimacs_missing_zero():
    """Case 4: missing terminating 0 should raise CNFError."""
    dimacs = "p cnf 2 1\n1 2"
    with pytest.raises(CNFError, match="Missing terminating 0"):
        read_dimacs_from_string(dimacs)

def test_dimacs_empty_clause_error():
    """Empty clauses should raise CNFError."""
    dimacs = "p cnf 1 1\n0"
    with pytest.raises(CNFError, match="Empty clause"):
        read_dimacs_from_string(dimacs)

def test_dimacs_multi_clause_per_line():
    """Multiple clauses on the same line."""
    dimacs = "p cnf 3 2\n1 2 0 2 3 0"
    doc = read_dimacs_from_string(dimacs)
    assert doc.clauses == [[1, 2], [2, 3]]

def test_dimacs_no_header_infer_vars():
    """If header missing, infer nvars."""
    dimacs = "1 -2 3 0\n-3 4 0"
    doc = read_dimacs_from_string(dimacs)
    assert doc.num_vars == 4
    assert doc.clauses == [[1, -2, 3], [-3, 4]]

def test_dimacs_header_smaller_than_actual():
    """If header says 2 but we see 3, use 3."""
    dimacs = "p cnf 2 1\n1 2 3 0"
    doc = read_dimacs_from_string(dimacs)
    assert doc.num_vars == 3
    assert doc.clauses == [[1, 2, 3]]
