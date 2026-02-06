import pytest
import shutil
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase

def test_db_roundtrip(tmp_path):
    db_root = tmp_path / "denabase_ut"
    db = DenaBase(str(db_root))
    
    # 1. Add IR Entry
    ir_data = [{"vars": ["a", "b"], "k": 1, "type": "exactly"}] # Minimal IR-like structure (list of dicts usually implies logic)
    # Actually add_ir expects compilable IR.
    # IR compiler handles list of Cardinality/BoolExpr.
    # Using python objects for simplicity if compiler supports dicts?
    # Compiler expects types from ir_types.
    
    from Denabase.Denabase.ir.ir_types import Exactly, VarRef
    
    ir_obj = Exactly(k=1, vars=[VarRef(name="a"), VarRef(name="b")])
    
    eid = db.add_ir(ir_obj, family="test", problem_id="p1", meta={"author": "me"})
    
    # 2. Check persistence
    assert (db_root / "entries" / f"{eid}.json").exists()
    assert (db_root / "cnf" / f"{eid}.cnf").exists()
    
    # 3. Reload DB
    db2 = DenaBase(str(db_root))
    rec = db2.store.get_entry_record(eid)
    
    assert rec is not None
    assert rec.meta.problem_id == "p1"
    assert rec.stats_summary["n_vars"] >= 2
    
    # Check artifact loading
    fp = db2.store.get_artifact(f"fingerprints/{eid}.json")
    assert fp is not None
    assert "content_hash" in fp
