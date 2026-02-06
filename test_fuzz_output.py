from engine.sat_manager import SATManager
from engine.state import AgentState, ModelingConstraint
import json

def test_fuzzing_output():
    mgr = SATManager()
    state = AgentState()
    state.active_ir_backend = "pb"
    
    # Setup some dummy vars
    state.sat_variables = {"x": 1, "y": 2}
    state.next_var_id = 3
    
    # Add a constraint
    c = ModelingConstraint(
        id="test_c",
        kind="exactly_one",
        parameters={"vars": ["x", "y"]}
    )
    state.model_constraints.append(c)
    
    # Run fuzzing
    print("Running fuzzing test...")
    payload = {
        "constraint_ids": ["test_c"],
        "num_tests": 20
    }
    mgr.fuzz_constraints(state, payload)

if __name__ == "__main__":
    test_fuzzing_output()
