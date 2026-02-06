
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from Denabase.agent.denabase_bridge import DenabaseBridge
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.trace import EncodingTrace, TraceEvent
from Denabase.Denabase.gadgets.gadget_registry import registry
from Denabase.Denabase.gadgets.macro_gadget import MacroGadget

def test_bridge_suggestions_ranking():
    # 1. Setup Temp DB
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "db"
    
    try:
        DenaBase.create(db_path)
        bridge = DenabaseBridge(str(db_path))
        
        # 2. Add Analogs with Traces
        # Entry A: Uses "at_most_k" frequently
        eid_a = bridge.db.add_ir([], "fam_a", "p1", verify=False)
        trace_a = EncodingTrace(problem_id="p1", family="fam_a", events=[
            TraceEvent(kind="IR_NODE", payload={"type": "at_most_k"}),
            TraceEvent(kind="IR_NODE", payload={"type": "at_most_k"}),
            TraceEvent(kind="IR_NODE", payload={"type": "at_most_k"})
        ])
        bridge.db.attach_trace(eid_a, trace_a)
        
        # Entry B: Uses "at_most_k" frequently too
        eid_b = bridge.db.add_ir([], "fam_a", "p2", verify=False)
        trace_b = EncodingTrace(problem_id="p2", family="fam_a", events=[
            TraceEvent(kind="IR_NODE", payload={"type": "at_most_k"}),
           TraceEvent(kind="IR_NODE", payload={"type": "at_most_k"})
        ])
        bridge.db.attach_trace(eid_b, trace_b)
        
        # Entry C: Unrelated (uses "linear_eq")
        eid_c = bridge.db.add_ir([], "fam_b", "p3", verify=False)
        trace_c = EncodingTrace(problem_id="p3", family="fam_b", events=[
            TraceEvent(kind="IR_NODE", payload={"type": "linear_eq"})
        ])
        bridge.db.attach_trace(eid_c, trace_c)
        
        # 3. Define Dummy Macros
        # Macro 1: Relevant (uses at_most_k)
        m1 = MacroGadget(name="RelevantMacro", description="desc", ir_template=[
            {"type": "at_most_k", "vars": "$v"}
        ], params_schema={})
        
        # Macro 2: Irrelevant
        m2 = MacroGadget(name="IrrelevantMacro", description="desc", ir_template=[
            {"type": "non_existent_primitive", "vars": "$v"}
        ], params_schema={})
        
        # Patch Registry to include our test macros
        # We save original state to restore later? But registry is global singleton.
        # Ideally we register and unregister.
        registry.register_macro(m1)
        registry.register_macro(m2)
        
        # 4. Mock Query to return specific favorites
        # We want our retrieval to hit entry A, B, C
        # Bridge calls db.query_similar
        with patch.object(bridge.db, 'query_similar') as mock_query:
            mock_query.return_value = [
                {"entry_id": eid_a, "final_score": 0.9, "problem_id": "p1", "family": "fam_a"},
                {"entry_id": eid_b, "final_score": 0.8, "problem_id": "p2", "family": "fam_a"},
                {"entry_id": eid_c, "final_score": 0.5, "problem_id": "p3", "family": "fam_b"}
            ]
            
            # 5. Call Retrieve
            priors = bridge.retrieve_priors("some query", top_k=3)
            
            # 6. Assertions
            macros = priors["suggested_macros"]
            motifs = priors["suggested_motifs"]
            
            # Motifs Check
            # "at_most_k" should be top motif. Count = 3 (A) + 2 (B) = 5
            top_motif = motifs[0]
            assert top_motif["type"] == "at_most_k"
            assert top_motif["count"] == 5
            
            # Macro Ranking Check
            # RelevantMacro should be ranked #1
            top_macro = macros[0]
            assert top_macro["name"] == "RelevantMacro"
            assert top_macro["score"] > 1.0 # Should have boost
            assert "matches analog motifs" in top_macro["reason"].lower()
            
            # Find Irrelevant
            irr = next((m for m in macros if m["name"] == "IrrelevantMacro"), None)
            if irr:
                assert irr["score"] == 1.0 # Base score only
                assert top_macro["score"] > irr["score"]

    finally:
        shutil.rmtree(tmp_dir)
        # Cleanup Registry
        if "RelevantMacro" in registry._macro_registry:
            del registry._macro_registry["RelevantMacro"]
        if "IrrelevantMacro" in registry._macro_registry:
            del registry._macro_registry["IrrelevantMacro"]
        pass 
