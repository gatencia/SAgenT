import pytest
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.cnf.cnf_types import CnfDocument

def test_verification_persistence(tmp_path):
    """
    Test that verification artifacts are persisted when verify=True,
    and are accessible via store.get_verification_record.
    """
    db_root = tmp_path / "db"
    db = DenaBase(str(db_root))
    
    # 1. Add IR with verify=True
    from Denabase.Denabase.ir.ir_types import And, Lit, VarRef
    ir_obj = And(terms=[Lit(var=VarRef(name="a")), Lit(var=VarRef(name="b"))])
    eid1 = db.add_ir(ir_obj, "test", "ir_prob", verify=True)
    
    # Check artifact existence on disk
    ver_path = db.store.dirs["verification"] / f"{eid1}.json"
    assert ver_path.exists()
    
    # Check retrieval
    rec1 = db.store.get_verification_record(eid1)
    assert rec1 is not None
    assert rec1.passed is True
    # Check config capture
    assert rec1.config["seconds_max"] == 2.0
    assert rec1.config["check_simplify_equisat"] is True
    # Check result details
    assert isinstance(rec1.checks_run, list)
    # "solve" is usually the first check
    assert "solve" in rec1.checks_run
    
    # 2. Add CNF with verify=True
    doc = CnfDocument(num_vars=2, clauses=[[1, 2], [-1, -2]])
    eid2 = db.add_cnf(doc, "test", "cnf_prob", verify=True)
    
    rec2 = db.store.get_verification_record(eid2)
    assert rec2 is not None
    assert rec2.passed is True
    assert rec2.config["num_metamorphic"] == 10
    
    # 3. Add CNF with verify=False (ensure no record)
    doc3 = CnfDocument(num_vars=1, clauses=[[1]])
    eid3 = db.add_cnf(doc3, "test", "no_check", verify=False)
    
    rec3 = db.store.get_verification_record(eid3)
    assert rec3 is None
    assert not (db.store.dirs["verification"] / f"{eid3}.json").exists()
