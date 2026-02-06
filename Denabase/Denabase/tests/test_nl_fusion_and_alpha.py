import os
import pytest
import json
import numpy as np
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ingest.satbench import ingest_manifest

# Force alpha model training threshold low
os.environ["DENABASE_ALPHA_MIN_EXAMPLES"] = "3"

def test_nl_fusion_and_alpha(tmp_path):
    """
    Verifies NL embeddings, fusion retrieval, and learned alpha.
    """
    db_root = tmp_path / "db"
    db = DenaBase(str(db_root))
    
    # 1. Ingest 3 cases
    # Case A: "Logic Puzzle Person" (SAT)
    # Case B: "Find one suspect" (Analogy to A)
    # Case C: "Graph coloring map" (Structurally different)
    
    # We create simple CNFs
    # A & B use p cnf 2 2 (structurally identical/isomorphic)
    # C uses p cnf 5 10 (distinct)
    
    manifest = [
        {
            "id": "case_a",
            "family": "logic_puzzle",
            "natural_language": "Exactly one person is guilty of the crime.",
            "label": "SAT",
            "cnf_dimacs": "p cnf 2 2\n1 2 0\n-1 -2 0\n" # Exactly 1 of 2
        },
        {
            "id": "case_b", 
            "family": "logic_puzzle",
            "natural_language": "Only one suspect did it.",
            "label": "SAT",
            "cnf_dimacs": "p cnf 2 2\n1 2 0\n-1 -2 0\n"
        },
        {
            "id": "case_c",
            "family": "coloring",
            "natural_language": "Graph coloring with two colors on a map.",
            "label": "UNSAT",
            "cnf_dimacs": "p cnf 3 3\n1 2 0\n2 3 0\n1 3 0\n" # cycle
        }
    ]
    
    # Ingest
    # Logic: add_satbench_case calls add_cnf -> updates ML
    for rec in manifest:
        db.add_satbench_case(
            dataset_id=rec["id"],
            family=rec["family"],
            problem_id=rec["id"],
            nl_text=rec["natural_language"],
            expected_label=rec["label"],
            cnf_dimacs=rec["cnf_dimacs"]
        )
        
    # Verify NL index exists
    assert db.nl_index_path.exists()
    assert db.alpha_model_path.exists()
    
    # Confirm alpha model fitted (we set threshold to 3)
    assert db.alpha_model.is_fitted
    
    # 2. Query Scenarios
    
    # Query with Case A's CNF
    from Denabase.Denabase.cnf.cnf_io import read_dimacs
    # Helper to parse string dimacs
    class DummyDoc:
        pass # add_satbench internal logic handled it. 
             # Here we need a CnfDocument for query.
    
    # Parse A's CNF
    # p cnf 2 2 ... 
    # Use internal parsing or helper
    from Denabase.Denabase.cnf.cnf_types import CnfDocument
    doc_a = CnfDocument(num_vars=2, clauses=[[1, 2], [-1, -2]])
    doc_c = CnfDocument(num_vars=3, clauses=[[1, 2], [2, 3], [1, 3]])
    
    # Scenario A: NL Dominant (alpha=0.0)
    # Query: "suspect" -> should match B high (and A) despite C
    res = db.query_similar(
        doc_c, 
        topk=3, 
        alpha=0.0, 
        nl_query_text="Find one suspect",
        use_learned_alpha=False
    )
    
    assert res[0]["problem_id"] == "case_b"
    assert res[0]["alpha_used"] == 0.0
    
    # Scenario B: Structure Dominant (alpha=1.0)
    # Query with doc_a. "suspect" text.
    # Should recall A and B (isomorphic) equally high on structure.
    # NL for "suspect" would favor B. 
    # But alpha=1.0 ignores NL.
    res = db.query_similar(
        doc_a,
        topk=3,
        alpha=1.0,
        nl_query_text="suspect",
        use_learned_alpha=False
    )
    
    ids = {r["problem_id"] for r in res[:2]}
    assert "case_a" in ids
    assert "case_b" in ids
    assert res[0]["alpha_used"] == 1.0
    
    # Scenario C: Learned Alpha
    # Query doc_a + "suspect".
    # Alpha model should predict something.
    res = db.query_similar(
        doc_a,
        topk=3,
        nl_query_text="suspect",
        use_learned_alpha=True
    )
    
    alpha = res[0]["alpha_used"]
    assert 0.0 <= alpha <= 1.0
    # Ideally, since it's "logic_puzzle" and small, alpha ~ 0.55 or 0.7
    # (Checking heuristic: logic_puzzle -> 0.55)
    
    # Check that alpha != 0.7 if family logic worked?
    # Logic in code: "If any(x in family ... logic ... ) => 0.55".
    # We used family="logic_puzzle".
    # So expected ~ 0.55.
    assert abs(alpha - 0.55) < 0.1
    
    # Output check
    assert "structural_similarity" in res[0]
    assert "nl_similarity" in res[0]
    assert "final_score" in res[0]

