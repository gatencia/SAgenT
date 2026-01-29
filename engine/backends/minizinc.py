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
from tools.mzn_to_fzn import compile_to_flatzinc, parse_flatzinc
from engine.booleanizer import Booleanizer
from engine.booleanizer import Booleanizer
from pysat.solvers import Solver
from engine.debug_harness import DebugHarness
import json

class MiniZincCoreBackend(IRBackend):
    def __init__(self):
        self.solver_id = "chuffed" # Used for FZN compilation compilation target if needed
        self.booleanizer = Booleanizer()

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

    def generate_code(self, state: AgentState) -> str:
        # 0. File-Centric Override
        if state.model_file_path and os.path.exists(state.model_file_path):
             try:
                 with open(state.model_file_path, "r") as f:
                     return f.read()
             except: pass

        # 1. Generate MZN
        mzn_lines = []
        
        # Variables (Boolean)
        max_id = state.next_var_id
        if state.sat_variables:
             max_id = max(max(state.sat_variables.values()) + 1, max_id)
        
        mzn_lines.append(f"array[1..{max_id}] of var bool: x;")
        
        # Helper to get MZN ref
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
                bools = [f"bool2int({v})" for v in vars_ref]
                mzn_lines.append(f"constraint sum([{', '.join(bools)}]) = 1;")
            elif k == "at_most_k":
                vars_ref = [get_mzn_ref(v) for v in p["vars"]]
                bools = [f"bool2int({v})" for v in vars_ref]
                mzn_lines.append(f"constraint sum([{', '.join(bools)}]) <= {p['k']};")
            # ... (Implement others as needed)
            
        # 2. Inject Raw Code (High-Level)
        if hasattr(state, "minizinc_code") and state.minizinc_code:
            mzn_lines.append("\n% --- User High-Level Code ---")
            mzn_lines.extend(state.minizinc_code)
            
        mzn_lines.append("solve satisfy;")
        return "\n".join(mzn_lines)

    def solve(self, state: AgentState) -> str:
        """
        Executes the MZN -> FZN -> CNF -> SAT pipeline with Debug Harness.
        """
        mzn_code = self.generate_code(state)
        
        
        # 1. Write Code to File
        # Check if we have an active file-based workflow
        if state.model_file_path and os.path.exists(state.model_file_path):
            f_mzn = state.model_file_path
            # We skip 'generate_code' and use this file directly
        else:
            # Fallback to generation
            mzn_code = self.generate_code(state)
            model_id = uuid.uuid4().hex[:8]
            f_mzn = f"memory/model_{model_id}.mzn"
            
            try:
                with open(f_mzn, "w") as f:
                    f.write(mzn_code)
            except Exception as e:
                return f"Write Error: {e}"
        
        try:
            # 2. Compile to FlatZinc
                
            # 2. Compile to FlatZinc
            try:
                fzn_content = compile_to_flatzinc(f_mzn)
            except Exception as e:
                return f"MiniZinc Compilation Failed (Syntax/Type Error):\n{e}"
                
            # 3. Parse and Translate to CNF
            try:
                vars_found, constrs_found = parse_flatzinc(fzn_content)
                
                local_booleanizer = Booleanizer()
                harness = DebugHarness(solver_name='g3')
                
                # Register Vars
                for v in vars_found:
                    if v['type'] == 'bool':
                        local_booleanizer.register_bool(v['name'])
                    # Int support pending
                
                # Reset Harness with correct max_id
                max_id = local_booleanizer.next_literal_id - 1
                harness.reset_instrumentation(max_id)
                
                # Translate Constraints & Instrument
                for c in constrs_found:
                    # We only handle bool_clause for now
                    if c['type'] == 'bool_clause':
                        args_str = c['args']
                        import re
                        m = re.match(r"\[(.*)\],\s*\[(.*)\]", args_str)
                        if m:
                            pos_part = m.group(1).split(',') if m.group(1).strip() else []
                            neg_part = m.group(2).split(',') if m.group(2).strip() else []
                            clause = []
                            for p in pos_part:
                                if p.strip(): clause.append(local_booleanizer.get_bool_literal(p.strip()))
                            for n in neg_part:
                                if n.strip(): clause.append(-local_booleanizer.get_bool_literal(n.strip()))
                            
                            # Add Group
                            # Group ID comes from mzn_to_fzn "group" field or fallback
                            gid = c.get("group", f"c_{uuid.uuid4().hex[:4]}")
                            harness.add_group(gid, [clause], {"raw": c["raw"], "type": c["type"]})
                    
                    elif c['type'] == 'bool_not':
                        # bool_not(a, b) => b <-> ~a => (b \/ a) /\ (~b \/ ~a)
                        args_part = c['args']
                        # Simple split, might need regex if complex args
                        parts = [x.strip() for x in args_part.split(',')]
                        if len(parts) >= 2:
                            a_name, b_name = parts[0], parts[1]
                            try:
                                lit_a = local_booleanizer.get_bool_literal(a_name)
                                lit_b = local_booleanizer.get_bool_literal(b_name)
                                
                                # (b \/ a)
                                c1 = [lit_b, lit_a]
                                # (~b \/ ~a)
                                c2 = [-lit_b, -lit_a]
                                
                                gid = c.get("group", f"c_{uuid.uuid4().hex[:4]}_not")
                                harness.add_group(gid, [c1, c2], {"raw": c["raw"], "type": c["type"]})
                            except ValueError as e:
                                print(f"Warning: Skipping bool_not on unknown vars: {e}")

                    else:
                        print(f"Warning: MiniZinc Backend ignoring unhandled FZN constraint: {c['type']} ({c['args']})")
                
                # 4. Diagnose using Harness
                report = harness.diagnose()
                
                # Write Debug Report
                with open("output_debug.json", "w") as f:
                    json.dump(report, f, indent=2)
                
                if report["status"] == "SAT":
                    # Retrieve the first model from harness if available or re-solve
                    # Ideally harness returns models. For now, let's re-solve simply to populate state
                    # or assume the harness left the solver in SAT state? No, harness reconstructs.
                    # Let's simple-solve for state population:
                    
                    solver = Solver(name="g3", bootstrap_with=harness.clauses) # Instrumented clauses!
                    solver.solve(assumptions=[g["selector"] for g in harness.groups.values()])
                    model = solver.get_model()
                    
                    decoded_vars = []
                    decoded_map = {}
                    for lit in model:
                        if lit > 0:
                             # Booleanizer map check
                             info = local_booleanizer.literal_to_info.get(lit)
                             if info:
                                 name = info["name"]
                                 decoded_vars.append(name)
                                 decoded_map[name] = True
                    
                    # PREPARE OBSERVATION DATA (Output to Agent, NOT file)
                    # The Agent is responsible for generating the final report.
                    
                    # 1. Variable Legend
                    sorted_vars = sorted(local_booleanizer.literal_to_info.items())
                    var_lookup = {lit: info['name'] for lit, info in sorted_vars}
                    legend_str = "\n".join([f"{lit} <-> {info['name']}" for lit, info in sorted_vars])

                    # 2. Readable CNF
                    clean_clauses = []
                    cnf_str_lines = []
                    for clause in harness.clauses:
                        clean = [l for l in clause if abs(l) <= max_id]
                        if clean:
                            clause_str_parts = []
                            for l in clean:
                                v_name = var_lookup.get(abs(l), f"var_{abs(l)}")
                                if l < 0:
                                    clause_str_parts.append(f"NOT {v_name}")
                                else:
                                    clause_str_parts.append(v_name)
                            cnf_str_lines.append(f"({' OR '.join(clause_str_parts)})")
                    cnf_str = "\n".join(cnf_str_lines)

                    # 3. Store Solution in State
                    state.solution = {v: (v in decoded_map) for v in local_booleanizer.var_map.keys() if local_booleanizer.var_map[v]["type"] == 'bool'}
                    solver.delete()
                    
                    # Return RICH observation
                    return (f"Solution Found!\n\n"
                            f"=== VARIABLE LEGEND ===\n{legend_str}\n\n"
                            f"=== FULL CNF CLAUSES ===\n{cnf_str}\n\n"
                            f"=== RAW SOLUTION ===\n{', '.join(decoded_vars)}\n\n"
                            f"INSTRUCTION: Use this information to write the final detailed report using the FINISH action.")
                
                else:
                    # UNSAT
                    # EXPORT CNF (User Request: Debug why CNF is wrong)
                    cnf_file = "output.cnf"
                    
                    # We need to build a temp solver to dump constraints, as we didn't keep one open.
                    # Harness has the clauses.
                    # dump_solver = Solver(name="g3", bootstrap_with=harness.clauses)
                    # dump_solver.to_file(cnf_file)
                    # dump_solver.delete()
                    
                    core_groups = report.get("minimized_core", [])
                    # Provide clearer feedback
                    reason_msg = f"UNSAT. The constraints are contradictory."
                    if core_groups:
                         reason_msg += f"\nConflict Core (Failed Constraints):\n"
                         for i, cg in enumerate(core_groups[:5]): # Show top 5
                             reason_msg += f"  - Group {cg}\n"
                         reason_msg += f"  (Total {len(core_groups)} conflicting groups. See output_debug.json for full details)"
                    return reason_msg

            except Exception as e:
                # import traceback
                # traceback.print_exc()
                return f"Translation/Harness Failed: {e}"

        except Exception as e:
            return f"Pipeline Error: {e}"
        finally:
            pass
