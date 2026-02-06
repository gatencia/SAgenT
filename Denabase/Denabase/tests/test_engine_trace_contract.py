
import pytest
import shutil
import tempfile
import json
from pathlib import Path

from engine.backends.pb import PBBackend
from engine.state import AgentState, ModelingConstraint
from engine.vars import VarManager
from Denabase.Denabase.trace import EncodingTrace, TraceEvent
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.induce.stitchlite import StitchLiteMiner

def test_trace_contract_compliance():
    # 1. Setup Backend & State
    backend = PBBackend()
    state = AgentState()
    # Mock Denabase trace
    state.denabase_trace = EncodingTrace(problem_id="test_p", family="test_f", events=[])
    state.var_manager = VarManager()
    state.var_manager.declare("x")
    state.var_manager.declare("y")
    state.var_manager.declare("z")
    state.sat_variables = {"x": 1, "y": 2, "z": 3}
    
    # 2. Add Constraints
    c1 = ModelingConstraint(id="c1", kind="exactly_one", parameters={"vars": ["x", "y"]}, ir_backend="pb")
    c2 = ModelingConstraint(id="c2", kind="at_most_k", parameters={"vars": ["x", "y", "z"], "k": 2}, ir_backend="pb")
    c3 = ModelingConstraint(id="c3", kind="implies", parameters={"a": "x", "b": "z"}, ir_backend="pb")
    
    state.model_constraints = [c1, c2, c3]
    
    # 3. Compile (triggers trace logging)
    backend.compile(state)
    
    events = state.denabase_trace.events
    assert len(events) == 3
    
    # 4. Assert Contract
    # Event 0: ExactlyOne
    e0 = events[0]
    p0 = e0.payload
    assert p0["type"] == "exactly_one"
    assert p0["arity"] == 2
    assert p0["k"] == 1
    assert "x" in p0["vars"]
    
    # Event 1: AtMostK
    e1 = events[1]
    p1 = e1.payload
    assert p1["type"] == "at_most_k"
    assert p1["arity"] == 3
    assert p1["k"] == 2
    
    # Event 2: Implies
    e2 = events[2]
    p2 = e2.payload
    assert p2["type"] == "implies"
    assert p2["arity"] == 2
    
def test_stitchlite_compatibility():
    # Verify that StitchLiteMiner can read these events without "Unknown" type
    # We can reuse the backend logic to generate a real trace payload
    
    # 1. Generate Events using Backend
    backend = PBBackend()
    state = AgentState()
    state.denabase_trace = EncodingTrace(problem_id="p", family="f", events=[])
    state.var_manager = VarManager()
    state.var_manager.declare("x")
    state.var_manager.declare("y")
    
    c = ModelingConstraint(id="c1", kind="exactly_one", parameters={"vars": ["x", "y"]}, ir_backend="pb")
    state.model_constraints = [c]
    backend.compile(state)
    
    event = state.denabase_trace.events[0]
    
    # 2. Test compatibility with StitchLiteMiner
    # We don't need a real DB for this unit test of the miner's abstraction logic
    miner = StitchLiteMiner(db=None) 
    
    # Check abstraction
    sig = miner._abstract_event(event.payload)
    # sig is (type, k, arity)
    assert sig[0] == "exactly_one"
    assert sig[0] != "Unknown"
    assert sig[1] == 1 # k
    assert sig[2] == 2 # arity
