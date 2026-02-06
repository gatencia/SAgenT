import pytest
import shutil
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.cnf.cnf_types import CnfDocument
from Denabase.Denabase.core.errors import VerificationError

@pytest.fixture
def tmp_db(tmp_path):
    d = tmp_path / "test_db_verify"
    return DenaBase(str(d))

def test_add_cnf_verify_fail_gate(tmp_db):
    """
    Test that if verification returns failure, ingestion aborts.
    We mock CnfVerifier to force a failure result.
    """
    from unittest.mock import patch, MagicMock
    from Denabase.Denabase.verify.verifier import VerificationResult
    
    doc = CnfDocument(num_vars=1, clauses=[[1]]) # Valid CNF
    
    with patch("Denabase.db.denabase.CnfVerifier") as MockVerifier:
        # Configure mock to return FAILED
        instance = MockVerifier.return_value
        instance.verify.return_value = VerificationResult(
            outcome="FAILED", 
            failures=["Simulated failure"]
        )
        
        with pytest.raises(VerificationError) as exc:
            tmp_db.add_cnf(doc, "fail", "mock_fail", verify=True)
            
        assert "Simulated failure" in str(exc.value)
    
    # Ensure no files written
    candidates = list(tmp_db.store.dirs["entries"].glob("*.json"))
    assert len(candidates) == 0

def test_add_cnf_verify_success(tmp_db):
    """
    Test valid SAT CNF passes verification.
    """
    doc = CnfDocument(num_vars=2, clauses=[[1, 2], [-1, -2]]) # SAT
    eid = tmp_db.add_cnf(doc, "ok", "valid", verify=True)
    assert eid
    assert (tmp_db.store.dirs["entries"] / f"{eid}.json").exists()

def test_add_cnf_no_verify(tmp_db):
    """
    Test standard ingestion with verification skipped (default for add_cnf? No default is False).
    """
    # use verify=False explicitly
    doc = CnfDocument(num_vars=2, clauses=[[1, 2]])
    eid = tmp_db.add_cnf(doc, "ok", "fast", verify=False)
    assert eid
    assert (tmp_db.store.dirs["entries"] / f"{eid}.json").exists()
