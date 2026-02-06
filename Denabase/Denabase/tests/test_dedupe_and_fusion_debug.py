import pytest
import shutil
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.cnf.cnf_types import CnfDocument

def test_dedupe_and_fusion(tmp_path):
    db_root = tmp_path / "test_db"
    db = DenaBase.create(str(db_root))
    
    # 1. Add duplicate entries (same dataset_id)
    doc1 = CnfDocument(num_vars=2, clauses=[[1, 2], [-1]])
    # Same content, different problem/entry IDs, but SAME dataset_id
    with db.bulk_ingest():
        db.add_satbench_case(
            dataset_id="test_case_1",
            family="logic",
            problem_id="p1",
            nl_text="The first entry about a logical puzzle.",
            expected_label="SAT",
            cnf_doc=doc1
        )
        
        db.add_satbench_case(
            dataset_id="test_case_1", # DUPLICATE case
            family="logic",
            problem_id="p2",
            nl_text="The second entry about the same puzzle.",
            expected_label="SAT",
            cnf_doc=doc1
        )
        
        # Add a distinct entry
        doc2 = CnfDocument(num_vars=3, clauses=[[1, 2, 3]])
        db.add_satbench_case(
            dataset_id="test_case_2",
            family="other",
            problem_id="p3",
            nl_text="Completely different scenario about space.",
            expected_label="SAT",
            cnf_doc=doc2
        )
    
    # Rebuild to ensure ML components are ready (though with only 3 it might be minimal)
    db.rebuild_nl_index()
    db.rebuild_index()
    
    # 2. Query and verify deduplication
    results = db.query_similar(doc1, topk=10, alpha=1.0)
    
    # Should only have 2 results total (one for test_case_1, one for test_case_2)
    # even though test_case_1 has two entries.
    assert len(results) == 2
    
    # Check dedupe keys
    keys = [r["dedupe_key"] for r in results]
    assert "satbench:test_case_1" in keys
    assert "satbench:test_case_2" in keys
    assert len(set(keys)) == len(keys)
    
    # 3. Verify component scores
    res0 = results[0]
    assert "final_score" in res0
    assert "structural_similarity" in res0
    assert "nl_similarity" in res0
    assert "alpha_used" in res0
    
    # 4. Verify NL channel
    # Query with text matching test_case_2
    results_nl = db.query_similar(None, nl_query_text="scenario about space", alpha=0.0)
    
    # test_case_2 should be #1
    assert results_nl[0]["dataset_id"] == "test_case_2"
    assert results_nl[0]["nl_similarity"] > 0
    
    # Verify alpha=0.5 fusion
    results_hybrid = db.query_similar(doc1, nl_query_text="scenario about space", alpha=0.5)
    # test_case_1 has perfect struct (1.0) but low NL (~0) -> score ~ 0.5
    # test_case_2 has low struct but high NL -> score ~ 0.5+
    
    print("\nHybrid results:")
    for r in results_hybrid:
        print(f"ID: {r['dataset_id']}, Final: {r['final_score']:.4f}, Struct: {r['structural_similarity']:.4f}, NL: {r['nl_similarity']:.4f}")

if __name__ == "__main__":
    pytest.main([__file__])
