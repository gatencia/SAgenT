import pytest
from Denabase.gadgets.gadget_registry import registry
from Denabase.verify.verifier import CnfVerifier, VerificationConfig
from Denabase.graph.fingerprints import compute_fingerprint
from Denabase.ir import compile_ir
from Denabase.cnf.cnf_types import CnfDocument

def test_registry_basics():
    gadgets = registry.list_gadgets()
    assert "exactly_one" in gadgets
    assert "k_coloring" in gadgets
    
    g = registry.get("exactly_one")
    assert g.name == "exactly_one"

def test_gadget_self_tests():
    # Use a real verifier config
    config = VerificationConfig(solver_name="m22", seconds_max=1.0)
    verifier = CnfVerifier(config)
    
    results = registry.run_self_tests(verifier)
    
    # Check that all tests passed for all gadgets
    for name, outcomes in results.items():
        assert all(outcomes), f"Gadget {name} failed self-tests: {outcomes}"

def test_gadget_fingerprint_stability():
    # Ensure that building a gadget twice yields identical fingerprint
    g = registry.get("k_coloring")
    params = {"n": 4, "k": 2, "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]} # Even cycle, 2-colorable? Yes.
    
    # Build 1
    ir1 = g.build_ir(params)
    clauses1, varmap1 = compile_ir(ir1)
    # Correct way to get max from list of lists
    if clauses1:
         max_var1 = max(abs(l) for c in clauses1 for l in c)
    else:
         max_var1 = len(varmap1)
         
    doc1 = CnfDocument(num_vars=max_var1, clauses=clauses1)
    fp1 = compute_fingerprint(doc1)
    
    # Build 2
    ir2 = g.build_ir(params)
    clauses2, varmap2 = compile_ir(ir2)
    
    if clauses2:
         max_var2 = max(abs(l) for c in clauses2 for l in c)
    else:
         max_var2 = len(varmap2)

    doc2 = CnfDocument(num_vars=max_var2, clauses=clauses2)
    fp2 = compute_fingerprint(doc2)
    
    assert fp1.content_hash == fp2.content_hash
    assert fp1.wl_hash == fp2.wl_hash
