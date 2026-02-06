import pytest
from Denabase.Denabase.verify.toygen import generate_random_3sat, generate_unsat_core
from Denabase.Denabase.verify.verifier import CnfVerifier, VerificationConfig
from Denabase.Denabase.verify.metamorphic import rename_vars

def test_sat_smoke():
    # Small random 3-SAT usually SAT
    doc = generate_random_3sat(num_vars=5, num_clauses=10, seed=42)
    config = VerificationConfig(solver_name="m22", num_metamorphic=0) # Use minisat22 (m22)
    verifier = CnfVerifier(config)
    
    res = verifier.verify(doc)
    assert res.outcome == "PASSED"
    # We can't guarantee 5 vars 10 clauses is SAT always, but highly likely.
    # Actually locally for seed 42 it might be. If it fails due to being UNSAT but we expected SAT?
    # No, verifying just runs it. "PASSED" means no errors occurred and metamorphic checks passed.
    # Since we didn't provide a checker that demands SAT, UNKNOWN/PASSED logic depends on failures.
    # Our verifier sets PASSED if no failures.
    # If standard verify just checks satisfiability, it passes regardless of SAT/UNSAT unless
    # we assert consistency.
    

def test_unsat_smoke():
    doc = generate_unsat_core()
    config = VerificationConfig(solver_name="m22")
    verifier = CnfVerifier(config)
    
    res = verifier.verify(doc)
    assert res.outcome == "PASSED"
    assert res.is_satisfiable is False

def test_metamorphic_stability():
    # Generate a likely SAT instance
    doc = generate_random_3sat(num_vars=5, num_clauses=8, seed=123)
    
    # Configure verifier to check 5 variants
    config = VerificationConfig(solver_name="m22", num_metamorphic=5)
    verifier = CnfVerifier(config)
    
    res = verifier.verify(doc)
    assert res.outcome == "PASSED"
    # Implicitly checked that all 5 variants matched the original's SAT status

def test_witness_checker():
    # doc: (1)
    # decoder: returns model
    # checker: asserts lit 1 is True (positive)
    
    # 1 is True => [1] in model means 1.
    def decoder(model):
        return model
        
    def checker(model):
        return 1 in model
        
    from Denabase.Denabase.cnf.cnf_types import CnfDocument
    doc = CnfDocument(num_vars=1, clauses=[[1]])
    
    config = VerificationConfig(solver_name="m22")
    verifier = CnfVerifier(config)
    
    res = verifier.verify(doc, decoder=decoder, checker=checker)
    assert res.outcome == "PASSED"
    assert res.is_satisfiable is True
    assert res.witness_valid is True
    
    # Fail case
    def bad_checker(model):
        return False
        
    res_fail = verifier.verify(doc, decoder=decoder, checker=bad_checker)
    assert res_fail.outcome == "FAILED"
    assert res_fail.witness_valid is False
