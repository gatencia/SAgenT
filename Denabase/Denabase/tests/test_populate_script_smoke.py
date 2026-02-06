import os
import json
import pytest
from pathlib import Path
from scripts import populate_denabase
from Denabase.Denabase.db.denabase import DenaBase

@pytest.fixture
def denabase_env(tmp_path):
    # Setup data dir
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # 1. Tiny CNF file
    cnf_file = data_dir / "test_scan.cnf"
    cnf_file.write_text("p cnf 1 1\n1 0\n")
    
    # 2. Tiny manifest
    manifest_file = tmp_path / "manifest.jsonl"
    recs = [
        {
            "id": "manifest_a",
            "family": "logic",
            "natural_language": "exactly one person",
            "label": "SAT",
            "cnf_dimacs": "p cnf 2 2\n1 2 0\n-1 -2 0\n"
        },
        {
            "id": "manifest_b",
            "family": "puzzle",
            "natural_language": "all red but one",
            "label": "UNSAT",
            "cnf_dimacs": "p cnf 2 2\n1 0\n2 0\n-1 -2 0\n"
        }
    ]
    with open(manifest_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
            
    db_root = tmp_path / "db"
    
    return {
        "db": db_root,
        "data": data_dir,
        "manifest": manifest_file
    }

def test_populate_script_smoke(denabase_env):
    # Set env to trigger alpha fit with 3 examples
    os.environ["DENABASE_ALPHA_MIN_EXAMPLES"] = "3"
    
    db_path = str(denabase_env["db"])
    data_path = str(denabase_env["data"])
    manifest_path = str(denabase_env["manifest"])
    
    # Call populate directly
    populate_denabase.main([
        "--db", db_path,
        "--data", data_path,
        "--manifest", manifest_path,
        "--rebuild-indexes",
        "--tags", "smoke,test"
    ])
    
    # 1. Check DB exists and contains entries
    assert os.path.exists(db_path)
    db = DenaBase.open(db_path)
    
    # total entries >= 5: 2 seeds + 2 manifest + 1 scanned
    # Actually if deduplication by content hash is active, and seeds/manifest overlap?
    # Our simple CNFs:
    # Seed 1: ExactlyOne(3 vars) -> distinct
    # Seed 2: AtMostOne(5 vars) -> distinct
    # Manifest A: ExactlyOne(2 vars) -> distinct from seeds
    # Manifest B: UNSAT cycle -> distinct
    # Scan: p cnf 1 1 -> distinct
    # All 5 should be distinct.
    
    assert db.total_entries >= 5
    
    # 2. NL Artifacts
    # In denabase.py:
    # self.nl_embedder_path = self.store.dirs["indexes"] / "nl_embedder.joblib"
    # self.nl_index_path = self.store.dirs["indexes"] / "nl_vector_index.bin"
    # Note: store.dirs["indexes"] is db_root / "indexes"
    
    idx_dir = Path(db_path) / "indexes"
    assert (idx_dir / "nl_embedder.joblib").exists()
    assert (idx_dir / "nl_vector_index.bin").exists()
    
    # 3. Alpha Model
    assert (idx_dir / "alpha_model.joblib").exists()
    assert db.alpha_model.is_fitted
    
    # 4. Query Sanity
    # Open DB, call query_similar with nl_query_text="exactly one" and alpha=0.0
    res = db.query_similar(
        # We need a query_obj (CNF/IR). We'll pass a dummy CNF.
        None, 
        topk=3,
        alpha=0.0,
        nl_query_text="exactly one person",
        use_learned_alpha=False
    )
    
    assert len(res) > 0
    # On tiny vocabulary, ranking can be noisy, but one of our manifest entries should be found
    names = [r["problem_id"] for r in res]
    assert any("manifest" in name for name in names)
    
    # Ensure result contains nl_similarity or alpha_used
    assert "nl_similarity" in res[0]
    assert "alpha_used" in res[0]
    assert res[0]["alpha_used"] == 0.0

if __name__ == "__main__":
    # For local debugging
    pytest.main([__file__])
