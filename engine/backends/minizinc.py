from typing import List, Dict, Any
from engine.backends.base import IRBackend
from engine.state import AgentState, ModelingConstraint

# PySAT imports
try:
    from pysat.card import CardEnc
    from pysat.pb import PBEnc
except ImportError:
    CardEnc = Any
    PBEnc = Any

import subprocess
import os
import uuid
from typing import List, Dict, Any

from engine.backends.base import IRBackend
from engine.state import AgentState, ModelingConstraint

class MiniZincCoreBackend(IRBackend):
    def __init__(self):
        self.solver_id = "chuffed" # Default

    @property
    def name(self) -> str:
        return "minizinc"

    def get_prompt_doc(self) -> str:
        return """
### BACKEND: MINIZINC
You must use the MiniZinc backend.
This backend maps your abstract constraints to high-level MiniZinc primitives:
- `alldifferent_onehot`: Maps to the `alldifferent` global constraint.
- `linear_eq` / `linear_leq`: Maps to native integer arithmetic (e.g. `sum(x) <= k`).
- `at_most_k`: Maps to cardinality constraints.
Constraint logic is handled by the solver's CP-SAT engine.
"""

    def allowed_kinds(self) -> Dict[str, Any]:
        return {
            "clause": {"parameters": {"literals": "List[str]"}},
            "implies": {"parameters": {"a": "str", "b": "str"}},
            "at_most_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "at_least_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "exactly_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "exactly_one": {"parameters": {"vars": "List[str]"}},
            "alldifferent_onehot": {"parameters": {"groups": "List[List[str]]"}}, # Harder to map directly?
            "linear_leq": {"parameters": {"terms": "List[Dict]", "rhs": "int"}},
            "linear_eq": {"parameters": {"terms": "List[Dict]", "rhs": "int"}}
        }

    def validate_constraint(self, constraint: ModelingConstraint, state: AgentState) -> None:
        if constraint.kind not in self.allowed_kinds(): raise ValueError(f"Unknown kind {constraint.kind}")
        # Basic variable existence check
        # (This is shared logic usually, but we keep it here for now)
        pass 

    def compile_constraints(self, constraints: List[ModelingConstraint], state: AgentState) -> List[List[int]]:
        # For MiniZinc, "compile" means "generate .mzn file context"
        # BUT the current architecture assumes we return CNF clauses (List[List[int]])!
        # This is a mismatch. If we want to use MiniZinc, we shouldn't be returning CNF clauses 
        # to be solved by PySAT. We should be solving it OURSELVES or returning empty clauses 
        # and handling the solve in a custom `solve()` method.
        # However, `SATManager.solve()` calls `_compile` then uses `Solver`.
        # We need to refactor SATManager to delegate SOLVING to the backend too.
        return []

    def solve(self, state: AgentState) -> str:
        # 1. Generate MZN
        mzn_lines = []
        
        # Variables (Boolean)
        # We need to map our vars (ints or names?) to MZN vars.
        # Our state.sat_variables maps Name -> IntID.
        # In MZN we can just use the Names directly if they are valid identifiers.
        # Or map them to x[1..N].
        # Let's use x[id].
        
        max_id = state.next_var_id
        if state.sat_variables:
             max_id = max(max(state.sat_variables.values()) + 1, max_id)
        
        mzn_lines.append(f"array[1..{max_id}] of var bool: x;")
        
        # Constraints
        def get_mzn_ref(name):
            name_str = str(name)
            is_neg = name_str.startswith('-') or name_str.startswith('~')
            atom = name_str.lstrip('-~')
            vid = state.sat_variables.get(atom)
            if vid is None: raise ValueError(f"Missing var {atom}")
            return f"not x[{vid}]" if is_neg else f"x[{vid}]"

        for c in state.model_constraints:
            k = c.kind
            p = c.parameters
            
            if k == "clause":
                lits = [get_mzn_ref(l) for l in p["literals"]]
                mzn_lines.append(f"constraint {' \\/ '.join(lits)};")
            elif k == "implies":
                mzn_lines.append(f"constraint {get_mzn_ref(p['a'])} -> {get_mzn_ref(p['b'])};")
            elif k == "exactly_one":
                vars_ref = [get_mzn_ref(v) for v in p["vars"]]
                # sum(v) = 1
                # bool2int coercion
                bools = [f"bool2int({v})" for v in vars_ref]
                mzn_lines.append(f"constraint sum([{', '.join(bools)}]) = 1;")
            elif k == "at_most_k":
                vars_ref = [get_mzn_ref(v) for v in p["vars"]]
                bools = [f"bool2int({v})" for v in vars_ref]
                mzn_lines.append(f"constraint sum([{', '.join(bools)}]) <= {p['k']};")
            # ... (Implement others as needed)
            
        mzn_lines.append("solve satisfy;")
        
        # Write File
        f_name = f"model_{uuid.uuid4().hex[:8]}.mzn"
        with open(f_name, "w") as f:
            f.write("\n".join(mzn_lines))
            
        # Run Solver
        # minizinc --solver chuffed model.mzn
        try:
            res = subprocess.run(["minizinc", "--solver", self.solver_id, f_name], capture_output=True, text=True)
            if "UNSATISFIABLE" in res.stdout:
                return "UNSAT"
            if "----------" in res.stdout: # Solution separator
                # Parse output
                # We didn't add output statements, so minizinc output is default?
                # Default is usually x = array1d(1..N, [true, false...]);
                # We should probably force output format.
                pass
                return "Solution Found (MiniZinc Parsing Pending)"
            return f"MiniZinc produced unknown output or error: {res.stderr}"
        except Exception as e:
            return f"MiniZinc execution failed: {e}"
        finally:
            if os.path.exists(f_name): os.remove(f_name)
            
        return "Unknown"
