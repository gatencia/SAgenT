import pytest
import math
from Denabase.profile.profile_types import (
    ConstraintProfile, profile_hash, profile_vector, profile_jaccard
)
from Denabase.profile.ir_profile import compute_ir_profile
from Denabase.profile.cnf_profile import compute_cnf_profile
from Denabase.ir.ir_types import VarRef, And, Or, Not, Lit, Exactly
from Denabase.cnf.cnf_types import CnfDocument

def test_profile_vector_shape():
    p = ConstraintProfile()
    vec = profile_vector(p)
    assert len(vec) == 16
    assert all(isinstance(x, float) for x in vec)

def test_profile_determinism():
    p1 = ConstraintProfile(counts={"and": 5}, cardinalities={"k": [1, 2]})
    p2 = ConstraintProfile(counts={"and": 5}, cardinalities={"k": [1, 2]})
    assert profile_hash(p1) == profile_hash(p2)
    assert profile_vector(p1) == profile_vector(p2)

def test_ir_renaming_invariance():
    # Expr 1: (a OR b)
    a1, b1 = VarRef(name="a"), VarRef(name="b")
    expr1 = Or(terms=[Lit(var=a1), Lit(var=b1)])
    
    # Expr 2: (x OR y) - structurally identical
    x1, y1 = VarRef(name="x"), VarRef(name="y")
    expr2 = Or(terms=[Lit(var=x1), Lit(var=y1)])
    
    p1 = compute_ir_profile(expr1)
    p2 = compute_ir_profile(expr2)
    
    assert profile_hash(p1) == profile_hash(p2)
    assert p1.counts == p2.counts

def test_cnf_profile_integration():
    doc = CnfDocument(num_vars=3, clauses=[[1, 2], [-1, 3]])
    p = compute_cnf_profile(doc)
    
    assert p.counts["clauses"] == 2
    assert p.counts["vars"] == 3
    assert p.cnf_stats["n_clauses"] == 2
    assert "density" in p.graphish

def test_profile_jaccard():
    p1 = ConstraintProfile(counts={"and": 10})
    p2 = ConstraintProfile(counts={"and": 10})
    assert math.isclose(profile_jaccard(p1, p2), 1.0)
    
    p3 = ConstraintProfile(counts={"or": 5})
    assert profile_jaccard(p1, p3) == 0.0 # No overlap in non-zero keys
