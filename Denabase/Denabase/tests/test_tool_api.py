import pytest
import json
from Denabase.Denabase.tooling.denabase_tool import DenabaseTool
from Denabase.Denabase.ir.ir_types import Exactly, VarRef

def test_tool_workflow(tmp_path):
    # Setup
    db_root = str(tmp_path / "db")
    tool = DenabaseTool(db_root)
    
    # 1. Add IR
    ir = Exactly(k=1, vars=[VarRef(name="a"), VarRef(name="b")])
    eid = tool.add_verified_ir(ir, "test_fam", "test_prob", {"author": "me"})
    assert eid
    
    # 2. Add DIMACS
    cnf_file = tmp_path / "test.cnf"
    cnf_file.write_text("p cnf 2 2\n1 2 0\n-1 -2 0\n")
    eid_cnf = tool.add_verified_dimacs(str(cnf_file), "test_fam", "test_cnf")
    assert eid_cnf
    
    # 3. Retrieve Similar
    # Query with the same IR should find the first entry
    matches = tool.retrieve_similar(ir, topk=5)
    assert len(matches) > 0
    # Ideally should match 'eid' (self-match)
    # Depending on embedding drift/indexing it might be close
    match_ids = [m["entry_id"] for m in matches]
    assert eid in match_ids
    
    # 4. Suggest Gadgets
    # (Won't find much since no provenance recorded yet, but shouldn't crash)
    suggestions = tool.suggest_gadgets(ir)
    assert "neighbor_gadgets" in suggestions
    assert isinstance(suggestions["neighbor_gadgets"], list)
    
    # 5. Recommend Encoding
    rec = tool.recommend_encoding_recipe(ir)
    assert "recipe" in rec
    assert rec["recipe"]["cardinality_encoding"] == "pairwise" # Small group size -> pairwise
