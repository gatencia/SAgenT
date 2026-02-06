import pytest
from Denabase.Denabase.autodoc.autodoc import AutoDoc
from Denabase.Denabase.ir.ir_types import Exactly, AtMost, VarRef, And, Lit

def test_autodoc_exactly_one():
    ir = Exactly(k=1, vars=[VarRef(name="a"), VarRef(name="b"), VarRef(name="c")])
    doc = AutoDoc().summarize(ir)
    
    # Requirements: must mention "exactly" and "one"
    desc = doc.description.lower()
    assert "exactly" in desc
    assert "one" in desc
    assert doc.category == "Cardinality/ExactlyOne"
    assert doc.params["n"] == 3

def test_autodoc_at_most_k():
    ir = AtMost(k=2, vars=[VarRef(name="x"), VarRef(name="y"), VarRef(name="z")])
    doc = AutoDoc().summarize(ir)
    
    assert doc.category == "Cardinality/AtMostK"
    assert "at most 2" in doc.description.lower()
    assert doc.params["k"] == 2

def test_autodoc_boolean():
    # Simple AND
    ir = And(terms=[
        Lit(kind="lit", var=VarRef(name="a")), 
        Lit(kind="lit", var=VarRef(name="b"))
    ])
    doc = AutoDoc().summarize(ir)
    assert "Boolean/Conjunction" in doc.category
