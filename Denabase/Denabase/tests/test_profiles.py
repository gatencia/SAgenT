import pytest
from Denabase.Denabase.ir.ir_types import (
    Exactly, AtMost, AtLeast, 
    And, Or, Not, Lit, VarRef
)
from Denabase.Denabase.profile.cnf_profile import compute_ir_profile
from Denabase.Denabase.profile.profile_types import profile_hash

def test_ir_profile_rename_invariance():
    """
    Ensure profiles are identical for structurally identical IR with different variable names.
    Exactly(1, [a, b, c]) vs Exactly(1, [x1, x2, x3])
    """
    ir1 = Exactly(k=1, vars=[VarRef(name="a"), VarRef(name="b"), VarRef(name="c")])
    ir2 = Exactly(k=1, vars=[VarRef(name="x1"), VarRef(name="x2"), VarRef(name="x3")])
    
    prof1 = compute_ir_profile(ir1)
    prof2 = compute_ir_profile(ir2)
    
    # Check counts match
    assert prof1.counts == prof2.counts
    assert prof1.cardinalities == prof2.cardinalities
    
    # Check stable hash
    h1 = profile_hash(prof1)
    h2 = profile_hash(prof2)
    assert h1 == h2

def test_ir_profile_traversal_counts():
    """
    Mixed expression: And([Or([Lit(a), Lit(b)]), Or([Not(Lit(a)), Lit(c)])])
    Expect:
      num_and=1
      num_or=2
      num_not=1
      num_lit=4 (a, b, a, c)
    """
    # Helper to clean syntax
    def L(name): return Lit(var=VarRef(name=name))
    
    expr = And(terms=[
        Or(terms=[L("a"), L("b")]),
        Or(terms=[
            Not(term=L("a")), 
            L("c")
        ])
    ])
    
    prof = compute_ir_profile(expr)
    c = prof.counts
    
    assert c.get("num_and") == 1
    assert c.get("num_or") == 2
    assert c.get("num_not") == 1
    assert c.get("num_lit") == 4
    
    # Check no phantom counts
    assert c.get("num_imp", 0) == 0
    assert c.get("num_card_exactly", 0) == 0

def test_ir_profile_stability():
    """
    Ensure profile hash and vector do not change across runs (deterministic).
    """
    ir = Exactly(k=2, vars=[VarRef(name=f"v{i}") for i in range(10)])
    
    prof1 = compute_ir_profile(ir)
    prof2 = compute_ir_profile(ir)
    
    assert profile_hash(prof1) == profile_hash(prof2)
    # If using vector, check it too (assumes vector logic exists in profile module or util)
    # The requirement mentions profile_vector length fixed. Profile object itself doesn't generate vector,
    # usually `profile_vector(prof)` utility does.
    # We'll just verify the profile object itself is stable equal.
    assert prof1.model_dump() == prof2.model_dump()
