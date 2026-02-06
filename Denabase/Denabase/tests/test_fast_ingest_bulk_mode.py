import pytest
import json
from pathlib import Path
from unittest.mock import patch
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.cnf.cnf_types import CnfDocument

def test_fast_ingest_bulk_mode(tmp_path):
    db_path = tmp_path / "bulk_db"
    db = DenaBase.create(str(db_path))
    
    # Create tiny manifest
    manifest_path = tmp_path / "mini_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for i in range(10):
            # p cnf 2 1 \n 1 2 0
            cnf_file = tmp_path / f"test_{i}.cnf"
            cnf_file.write_text(f"p cnf 2 1\n1 2 0")
            
            rec = {
                "dataset_id": "test",
                "family": "smoke",
                "problem_id": f"p_{i}",
                "description": f"Description for {i}",
                "label": "SAT",
                "cnf_path": str(cnf_file)
            }
            f.write(json.dumps(rec) + "\n")

    # Mock rebuild methods to count calls
    with patch.object(DenaBase, 'rebuild_index', wraps=db.rebuild_index) as mock_struct, \
         patch.object(DenaBase, 'rebuild_nl_index', wraps=db.rebuild_nl_index) as mock_nl, \
         patch.object(DenaBase, 'rebuild_alpha_model', wraps=db.rebuild_alpha_model) as mock_alpha:
        
        # We also need to avoid _update_ml_components being called
        # Actually in bulk mode it's skipped.
        
        # Run bulk ingest
        with db.bulk_ingest(rebuild=True):
            for i in range(10):
                doc = CnfDocument(num_vars=2, clauses=[[1, 2]])
                texts = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Iced", "Jackfruit"]
                db.add_cnf(doc, family="smoke", problem_id=f"bulk_{i}", nl_text=f"This is a {texts[i]} problem")
                
                # Verify rebuilds NOT called yet
                assert mock_struct.call_count == 0
                assert mock_nl.call_count == 0
                assert mock_alpha.call_count == 0
        
        # After exit, rebuilds should be called exactly once
        # Note: rebuild_alpha_model only runs if enough examples (min_ex)
        # We'll set environmental variable for min examples
        import os
        os.environ["DENABASE_ALPHA_MIN_EXAMPLES"] = "1"
        
        # Wait, rebuild_alpha_model check is in end_bulk_ingest
        # It was ALREADY called when exiting the context manager above.
        
        # But wait, mock_alpha was wrapped AFTER db was initialized? 
        # No, patch.object works on the instance if provided.
        
        assert mock_struct.call_count == 1
        assert mock_nl.call_count == 1
        # Alpha depends on dirty flag. add_cnf sets it.
        # But if total < min_ex, rebuild_alpha_model might skip?
        # end_bulk_ingest calls rebuild_alpha_model directly if dirty.
        # rebuild_alpha_model itself checks for examples.
        assert mock_alpha.call_count == 1

    # Final check: query works
    results = db.query_similar(None, nl_query_text="Fig", topk=5)
    assert len(results) > 0
    # One of them should be "bulk_5" roughly if NL works
    found = any(r["problem_id"] == "bulk_5" for r in results)
    # With 10 items and simple TFIDF, it should find it.
    assert found

def test_fast_ingest_dirty_flags(tmp_path):
    db_path = tmp_path / "dirty_db"
    db = DenaBase.create(str(db_path))
    
    with db.bulk_ingest(rebuild=False):
        db.add_cnf(CnfDocument(num_vars=2, clauses=[[1]]), family="f", problem_id="p")
        
    meta = db.db_meta
    assert meta.structural_index_dirty == True
    assert meta.nl_index_dirty == True
    assert meta.alpha_model_dirty == True
    
    # Manually rebuild one
    db.rebuild_nl_index()
    meta2 = db.db_meta
    assert meta2.nl_index_dirty == False
    assert meta2.structural_index_dirty == True
