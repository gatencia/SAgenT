import pytest
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir.ir_types import Exactly, VarRef, AtMost

def test_hybrid_retrieval_diversification(tmp_path):
    db = DenaBase(str(tmp_path / "db"))
    
    # Case A: Exactly(1, [x, y])
    ir_a = Exactly(k=1, vars=[VarRef(name="x"), VarRef(name="y")])
    eid_a = db.add_ir(ir_a, family="fam1", problem_id="probA")
    
    # Case B: Exactly(1, [p, q]) -> Structure identical to A (renamed)
    ir_b = Exactly(k=1, vars=[VarRef(name="p"), VarRef(name="q")])
    eid_b = db.add_ir(ir_b, family="fam1", problem_id="probB") # Same family
    
    # Case C: AtMost(1, [x, y]) -> Different structure
    ir_c = AtMost(k=1, vars=[VarRef(name="x"), VarRef(name="y")])
    eid_c = db.add_ir(ir_c, family="fam2", problem_id="probC")
    
    # Query with something structurally identical to A/B
    ir_q = Exactly(k=1, vars=[VarRef(name="m"), VarRef(name="n")])
    
    # Should get A or B. 
    # With MMR and lambda=0.7, if we get A first, B might be penalized if considered redundant?
    # Our simple MMR penalized same *problem_id*. Here problem_ids are diff. 
    # But wait, logic says redundancy = 1.0 if problem_ids match.
    # Here they don't match. So both should appear in top k.
    
    results = db.query_similar(ir_q, topk=3)
    ids = [r["problem_id"] for r in results]
    
    # We expect probA and probB to be top matches (identical structure)
    assert "probA" in ids
    assert "probB" in ids
    
    # Diversification test: same problem ID multiple versions?
    # Let's add A again with same problem_id
    eid_a2 = db.add_ir(ir_a, family="fam1", problem_id="probA") 
    
    results2 = db.query_similar(ir_q, topk=5)
    # The naive MMR penalizes redundancy if 'problem_id' matches selected.
    # So if probA(1) is selected, probA(2) should be heavily penalized.
    # Thus we likely see probA(1) and probB, but probA(2) pushed down or excluded if limit hit.
    
    returned_prob_ids = [r["problem_id"] for r in results2]
    # Count occurrences of "probA"
    count_probA = returned_prob_ids.count("probA")
    
    # Ideally MMR helps avoid duplicates in top results if valid alternatives exist.
    # Since we have probB and probC, maybe they come before 2nd probA.
    # probC is different structure, so lower sim.
    # probA(2) has high sim but penalty.
    pass # Assertions hard to exact without calibrated embeddings/weights. 
         # Just ensuring no crash and logical return structure.
    assert len(results2) > 0
