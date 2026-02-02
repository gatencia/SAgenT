import shutil
import tempfile
import pytest
from pathlib import Path
from engine.agent import ReActAgent
from engine.state import AgentState, ModelingConstraint
from engine.actions import ActionType
from Denabase.db.denabase import DenaBase
from Denabase.ir import Exactly, VarRef
from agent.denabase_bridge import DenabaseBridge

def test_agent_integration_full_flow():
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "denabase_db"
    
    try:
        # 1. Setup DB
        db = DenaBase.create(db_path)
        # Add seed data
        db.add_ir(Exactly(k=1, vars=[VarRef(name="x")]), "seed_family", "seed_p1", 
                  nl_text="A problem with exactly one variable", verify=False)
        db.rebuild_nl_index()
        
        # 2. Reset Singleton Bridge to use temp DB
        # This is a bit hacky but needed since Agent uses singleton
        DenabaseBridge._instance = None 
        bridge = DenabaseBridge(db_path)
        DenabaseBridge._instance = bridge
        
        # 3. Init Agent
        # Mock LLM calling since we test integration logic
        def mock_llm(prompt):
            return '{"action": "FINISH", "action_input": "Done"}'
            
        agent = ReActAgent(llm_callable=mock_llm)
        
        # 4. Verify Retrieval in Run
        # We manually verify usage since Mock LLM ignores prompt
        priors = bridge.retrieve_priors("Need exactly one variable")
        assert len(priors["analogs"]) > 0
        
        # 5. Verify Trace Capture in Solve
        # We construct a state manually to simulate "Ready to Solve"
        state = AgentState()
        state.active_ir_backend = "pb"
        state.sat_variables = {"x": 1, "y": 2} # Legacy map
        state.var_manager.declare("x")
        state.var_manager.declare("y")
        
        # Add constraint: ExactlyOne(x, y)
        c = ModelingConstraint(
            id="c1", 
            ir_backend="pb", 
            kind="exactly_one", 
            parameters={"vars": ["x", "y"]}
        )
        state.model_constraints.append(c)
        
        # Solve
        res = agent.sat.solve(state)
        
        # Assertions
        assert "Solution Found (SAT)" in res
        assert state.denabase_trace is not None
        assert len(state.denabase_trace.events) > 0
        assert state.denabase_trace.events[0].payload["kind"] == "exactly_one"
        
        # Verify Trace was stored in DB
        # The ID is random, so check count
        entries = [e for e in db.store.iter_entry_metas() if e.family == "agent_auto"]
        assert len(entries) == 1
        eid = entries[0].entry_id
        
        # Check attached trace
        t_path = db.root / "traces" / f"{eid}.json"
        assert t_path.exists()
        
    finally:
        shutil.rmtree(tmp_dir)
        DenabaseBridge._instance = None
