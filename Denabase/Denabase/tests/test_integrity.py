import pytest
import json
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir.ir_types import Exactly, VarRef

def test_integrity_scan_and_repair(tmp_path):
    db = DenaBase(str(tmp_path / "db"))
    
    # 1. Create a valid entry
    ir_obj = Exactly(k=1, vars=[VarRef(name="a"), VarRef(name="b")])
    eid = db.add_ir(ir_obj, family="test", problem_id="clean")
    
    # Check initial clean state
    report = db.scan_integrity()
    assert report.is_clean()
    assert len(report.valid_entries) == 1
    
    # 2. Corrupt the fingerprint file
    # We need to find where it is
    fp_path = db.root / "fingerprints" / f"{eid}.json"
    assert fp_path.exists()
    
    # Truncate it to make invalid JSON
    with open(fp_path, "w") as f:
        f.write("{ invalid json")
        
    # 3. Check detection
    report_bad = db.scan_integrity()
    assert not report_bad.is_clean()
    assert len(report_bad.corrupt_json) > 0
    assert any("fingerprints" in s for s in report_bad.corrupt_json)
    
    # 4. Corrupt logical content (fingerprint mismatch)
    # Restore valid JSON but with wrong hash. 
    # Use valid schema!
    
    fake_fp = {
        "signature_key": "wrong",
        "wl_hash": "wrong",
        "content_hash": "wrong",
        "lcg_hash": "wrong",
        "invariants": {},
        "feature_vector": []
    }
    with open(fp_path, "w") as f:
        json.dump(fake_fp, f)
        
    report_fake = db.scan_integrity()
    assert not report_fake.is_clean()
    assert len(report_fake.fingerprint_mismatches) > 0
    
    # 5. Repair
    # Logic: repair_indexes only indexes VALID entries.
    # Since this entry is now invalid (falsified FP), it should NOT be indexed.
    # So index count should be 0.
    
    count = db.rebuild_index()
    assert count == 0
    
    # 6. Restore to valid state manually (to prove repair works when valid)
    # We re-add the same problem to get fresh artifacts
    # Actually add_ir might overwrite or fail if ID collision?
    # IDs are UUIDs, so new add = new ID.
    # Let's ignore old broken one and add new one.
    
    eid2 = db.add_ir(ir_obj, family="test", problem_id="clean_again")
    
    # Now valid=1 (eid2), invalid=1 (eid)
    count_rec = db.rebuild_index()
    assert count_rec == 1
    
    # Verify vector index has 1 item
    # Internal list of ids access (if exposed, else query)
    assert len(db.index.ids) == 1
    assert db.index.ids[0] == eid2
