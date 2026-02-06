import pytest
from Denabase.Denabase.opt.metrics import score_candidate
from Denabase.Denabase.verify.verifier import VerificationResult
from Denabase.Denabase.opt.search import EncodingSearcher
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir.ir_types import Exactly, VarRef

def test_score_candidate():
    res = VerificationResult(outcome="PASSED", stats={"duration": 1.0})
    s = score_candidate(res)
    assert s > 0
    assert s == 50.0 # 100 / 2
    
    res_fail = VerificationResult(outcome="FAILED")
    assert score_candidate(res_fail) == 0.0

def test_searcher_smoke(tmp_path):
    db = DenaBase(str(tmp_path))
    searcher = EncodingSearcher(db)
    
    ir = Exactly(k=1, vars=[VarRef(name="a"), VarRef(name="b")])
    
    best = searcher.search(ir, time_budget=2.0)
    
    assert best is not None
    assert best.is_correct
    assert "encoding_recipe" in best.model_dump()
    assert "solver_config" in best.model_dump()
