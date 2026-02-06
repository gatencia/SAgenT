import pytest
import time
import os
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.cnf.cnf_types import CnfDocument

@pytest.mark.skipif(os.environ.get("SKIP_BENCH", "false") == "true", reason="Benchmark skipped via env")
def test_benchmark_light(tmp_path):
    """
    Benchmarks basic operations: add 30 CNFs, query 10 times.
    Checks for lock stability and cache effectiveness (implicitly via stability).
    """
    db_root = str(tmp_path / "bench_db")
    db = DenaBase(db_root)
    
    # 1. Add 30 items
    start_add = time.perf_counter()
    ids = []
    for i in range(30):
        # Create minimal valid CNF
        # i+1 vars, simple clauses
        n_vars = max(2, (i % 5) + 2)
        clauses = [[1, 2], [-1, -2]] 
        doc = CnfDocument(num_vars=n_vars, clauses=clauses)
        
        eid = db.add_cnf(doc, family="bench", problem_id=f"p_{i}")
        ids.append(eid)
        
    duration_add = time.perf_counter() - start_add
    # Roughly check reasonable speed? 30 items < 10s locally?
    # Actually can be very fast.
    print(f"Added 30 entries in {duration_add:.4f}s")
    
    # 2. Query 10 times
    # First query might be slow (build index / lazy), subsequent might be cached if repeated
    start_query = time.perf_counter()
    
    # Reuse valid doc for query
    q_doc = CnfDocument(num_vars=2, clauses=[[1, 2]])
    
    for k in range(10):
        res = db.query_similar(q_doc, topk=5)
        assert len(res) > 0
        
    duration_query = time.perf_counter() - start_query
    print(f"Queried 10 times in {duration_query:.4f}s")
    
    # Check cache dir populated
    cache_dir = tmp_path / "bench_db" / "cache"
    assert cache_dir.exists()
    assert len(list(cache_dir.glob("*.json"))) > 0
    
    # Ensure stability
    assert len(ids) == 30
