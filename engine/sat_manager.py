import copy
import json
import uuid
import random
from typing import List, Dict, Any, Optional

from engine.state import AgentState, ModelingConstraint, AgentPhase
from engine.backends.registry import IRBackendRegistry


# PySAT imports
try:
    from pysat.solvers import Solver
except ImportError:
    Solver = Any

# ANSI Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

class SATManager:
    def __init__(self, registry: Optional[IRBackendRegistry] = None):
        self.solver_name = 'g3'
        self.registry = registry if registry else IRBackendRegistry()

    def _compile(self, state: AgentState):
        """JIT compilation with Telemetry."""
        backend = self.registry.get(state.active_ir_backend)
        valid = []
        for c in state.model_constraints:
            if c.ir_backend != state.active_ir_backend:
                raise ValueError(f"Mixed backend constraint {c.id}")
            backend.validate_constraint(c, state)
            valid.append(c)
        
        # Incremental Compilation for Telemetry
        clauses = []
        report = {
            "num_user_vars": len(state.sat_variables),
            "max_var_id": state.next_var_id,
            "num_aux_vars": 0,
            "num_clauses": 0,
            "clauses_by_constraint_id": {},
            "aux_vars_by_constraint_id": {} # Not easily tracked with PySAT encodings but placeholders
        }

        # Deterministic Sort
        valid.sort(key=lambda x: x.id)

        current_max = state.next_var_id
        if state.sat_variables:
             current_max = max(max(state.sat_variables.values()) + 1, current_max)

        # Baseline aux var start
        start_aux = current_max

        # We must compile cumulatively or restart var pools? 
        # PySAT encoders usually take `top_id`. 
        # For simplicity AND correctness, we will just measure clause diffs. 
        # We assume compile_constraints handles list. Ideally we'd compile one by one?
        # But 'alldifferent' etc might be optimized in groups. 
        # Let's try one-by-one for telemetry, accepting potential optimization loss if backend allows.
        # PBBackend compiles one-by-one mostly.

        for c in valid:
             # Capture state before
             prev_count = len(clauses)
             
             # Compile single
             # NOTE: compile_constraints expects a list.
             new_c_clauses = backend.compile_constraints([c], state)
             
             # Update IDs? 
             # The backend uses `state.next_var_id` or similar. 
             # We need to make sure we don't overlap IDs if we just loop.
             # Actually `compile_constraints` logic inside PBBackend uses `top_id` from state.
             # We need to update state.next_var_id as we go to simulate cumulative build.
             
             # Wait, `compile_constraints` implementation re-reads `top_id` from state every time?
             # Let's look at PBBackend: it determines `top_id` from `state.next_var_id - 1`.
             # AND it calculates `top_id` from max literal in produced clauses.
             # BUT it does NOT update `state.next_var_id`. 
             # So if we call it in a loop without updating state, we reuse IDs!
             
             # FIX: We need to manually update state.next_var_id between calls for telemetry to work correctly.
             # But this modifies state! `_compile` is supposed to modify state. So this is fine.
             
             clauses.extend(new_c_clauses)
             
             # Update watermark
             local_max = start_aux
             for cl in new_c_clauses:
                 for l in cl:
                     local_max = max(local_max, abs(l))
             
             state.next_var_id = local_max + 1
             
             count = len(new_c_clauses)
             report["clauses_by_constraint_id"][c.id] = count
        
        state.cnf_clauses = clauses
        report["num_clauses"] = len(clauses)
        report["num_aux_vars"] = state.next_var_id - start_aux
        report["max_var_id"] = state.next_var_id
        
        state.compile_report = report

        # Sanity Guard
        for i, c in enumerate(state.cnf_clauses):
            if not c: raise ValueError(f"Empty clause produced at index {i}")
            for l in c:
                if l == 0: raise ValueError(f"Zero literal in clause {i}")

    def compile_subset(self, state: AgentState, subset: List[ModelingConstraint]) -> List[List[int]]:
        """Helper to compile a subset of constraints without modifying state."""
        # Deep isolation
        temp_state = copy.copy(state)
        temp_state.cnf_clauses = []
        
        # Re-compute watermark to ensure no collision with existing CNF vars if any
        max_used = 0
        if state.sat_variables:
            max_used = max(state.sat_variables.values())
        for c in state.cnf_clauses: # Check ORIGINAL clauses for high water mark
            for l in c:
                max_used = max(max_used, abs(l))
        temp_state.next_var_id = max_used + 1
        
        backend = self.registry.get(state.active_ir_backend)
        for c in subset:
             if c.ir_backend != state.active_ir_backend:
                 raise ValueError("Backend mismatch in subset")
             try: 
                 backend.validate_constraint(c, temp_state)
             except Exception as e:
                 raise ValueError(f"Invalid subset constraint {c.id} ({c.kind}): {e}")
        
        return backend.compile_constraints(subset, temp_state)

    def solve_cnf_under_assumptions(self, cnf_clauses: List[List[int]], assumptions: List[str], state: AgentState) -> str:
        assumption_ints = []
        for lit in assumptions:
            is_neg = str(lit).startswith('-') or str(lit).startswith('~')
            atom = str(lit).lstrip('-~')
            if atom not in state.sat_variables: return "ERROR_VAR"
            vid = state.sat_variables[atom]
            assumption_ints.append(-vid if is_neg else vid)
        
        try:
            solver = Solver(name=self.solver_name, bootstrap_with=cnf_clauses)
            sat = solver.solve(assumptions=assumption_ints)
            solver.delete()
            return "SAT" if sat else "UNSAT"
        except Exception as e:
            return f"ERROR:{str(e)}"

    def fuzz_constraints(self, state: AgentState, payload: Dict[str, Any]) -> str:
        c_ids = payload.get("constraint_ids", [])
        num_tests = payload.get("num_tests", 10)
        mode = payload.get("mode", "both")
        
        results = {"tested": 0, "total_tests": 0, "failures": 0, "failure_examples": []}
        
        # Map ID to constraint
        c_map = {c.id: c for c in state.model_constraints}
        
        # Literal Helpers
        def neg(lit: str) -> str:
             if lit.startswith('-'): return lit[1:]
             if lit.startswith('~'): return lit[1:]
             return f"-{lit}"

        def assert_lit(lit: str, truth: bool) -> str:
             return lit if truth else neg(lit)

        for cid in c_ids:
            if cid not in c_map: continue
            constr = c_map[cid]
            results["tested"] += 1
            
            # Setup Subset
            subset = [constr]
            # Heuristic background inclusion for alldifferent
            if constr.kind == "alldifferent_onehot":
                groups = constr.parameters["groups"]
                # For each group, try to find an exactly_one constraint defined on it
                for g in groups:
                    g_set = set(g)
                    for other in state.model_constraints:
                        if other.kind == "exactly_one" and set(other.parameters.get("vars", [])) == g_set:
                            subset.append(other)
            
            try:
                cnf = self.compile_subset(state, subset)
            except Exception as e:
                # If compilation fails, log it
                results["failures"] += 1
                results["failure_examples"].append({"id": cid, "error": str(e)})
                continue

            # Deterministic RNG
            rng = random.Random(cid)
            
            # Generate Tests
            targets = [] # List of (assumptions, expected_outcome, metadata)
            
            def gen_tests(n, positive: bool):
                generated = [] # List of (assumptions, metadata)
                p = constr.parameters
                k = constr.kind
                
                for _ in range(n):
                    meta = {}
                    assumps = []
                    valid_gen = False

                    if k == "implies":
                         # implies(a,b). Pos: -a OR (a,b). Neg: (a, -b)
                         a, b = p["a"], p["b"]
                         if positive:
                             if rng.choice([True, False]): # -a
                                 assumps = [neg(a)]
                             else: # a, b
                                 assumps = [a, b]
                         else: # a, -b
                             assumps = [a, neg(b)]
                         valid_gen = True

                    elif k in ["at_most_k", "at_least_k", "exactly_k", "exactly_one"]:
                        vs = p["vars"]
                        target_k = p.get("k", 1)
                        if k == "exactly_one": target_k = 1
                        
                        # Helpers based on indices
                        if positive:
                             # Try to satisfy
                             if target_k <= len(vs) and target_k >= 0:
                                 # Pick exactly target_k true
                                 true_idxs = set(rng.sample(range(len(vs)), target_k))
                                 assumps = [assert_lit(v, i in true_idxs) for i,v in enumerate(vs)]
                                 valid_gen = True
                        else:
                             # Try to violate
                             options = []
                             if k in ["at_most_k", "exactly_k", "exactly_one"] and target_k < len(vs):
                                 options.append(target_k + 1)
                             if k in ["at_least_k", "exactly_k", "exactly_one"] and target_k > 0:
                                 options.append(target_k - 1)
                             
                             if options:
                                 fk = rng.choice(options)
                                 true_idxs = set(rng.sample(range(len(vs)), fk))
                                 assumps = [assert_lit(v, i in true_idxs) for i,v in enumerate(vs)]
                                 valid_gen = True

                    elif k in ["linear_leq", "linear_eq"]:
                        terms = p["terms"]
                        rhs = int(p["rhs"])
                        
                        # 1. Unique Atoms
                        atoms = set()
                        for t in terms: 
                            atoms.add(str(t["var"]).lstrip('-~'))
                        
                        # 2. Assign Atoms
                        atom_vals = {a: rng.choice([True, False]) for a in atoms}
                        
                        # 3. Compute LHS
                        lhs_val = 0
                        for t in terms:
                            v = str(t["var"])
                            coef = int(t["coef"])
                            atom = v.lstrip('-~')
                            is_neg = v.startswith('-') or v.startswith('~')
                            
                            # Truth of literal
                            a_val = atom_vals[atom]
                            lit_true = a_val if not is_neg else (not a_val)
                            
                            if lit_true: lhs_val += coef
                            
                        # 4. Check Math
                        math_sat = (lhs_val <= rhs) if k == "linear_leq" else (lhs_val == rhs)
                        
                        if math_sat == positive:
                            assumps = [assert_lit(a, tv) for a, tv in atom_vals.items()]
                            meta = {"lhs": lhs_val, "rhs": rhs, "assign": atom_vals}
                            valid_gen = True

                    elif k == "alldifferent_onehot":
                         groups = p["groups"]
                         if positive:
                             # Distinct indices for each group
                             g_len = len(groups[0])
                             if len(groups) <= g_len:
                                 idxs = rng.sample(range(g_len), len(groups))
                                 assumps = []
                                 for g_i, val_i in enumerate(idxs):
                                     for bit_i, v in enumerate(groups[g_i]):
                                         assumps.append(assert_lit(v, bit_i == val_i))
                                 valid_gen = True
                         else:
                             # Collision: two groups same index
                             if len(groups) >= 2:
                                 g_idxs = rng.sample(range(len(groups)), 2)
                                 val_idx = rng.randint(0, len(groups[0])-1)
                                 g1, g2 = groups[g_idxs[0]], groups[g_idxs[1]]
                                 
                                 assumps = []
                                 # 1. Force collision
                                 assumps.append(g1[val_idx])
                                 assumps.append(g2[val_idx])
                                 
                                 # 2. Force all other bits in these two groups to False
                                 for bit_i in range(len(groups[0])):
                                     if bit_i != val_idx:
                                         assumps.append(assert_lit(g1[bit_i], False))
                                         assumps.append(assert_lit(g2[bit_i], False))
                                 valid_gen = True
                    
                    if valid_gen:
                        generated.append((assumps, meta))
                
                return generated

            if mode in ["both", "positive"]:
                for t, m in gen_tests(num_tests, True): targets.append((t, "SAT", m))
            if mode in ["both", "negative"]:
                for t, m in gen_tests(num_tests, False): targets.append((t, "UNSAT", m))
            
            results["total_tests"] += len(targets)
            
            # Execute
            fail_rec = []
            for assump, expected, meta in targets:
                outcome = self.solve_cnf_under_assumptions(cnf, assump, state)
                if outcome != expected:
                    results["failures"] += 1
                    rec = {
                        "id": cid, 
                        "kind": constr.kind, 
                        "backend": state.active_ir_backend,
                        "assumed": assump, 
                        "expected": expected, 
                        "got": outcome,
                        "debug": meta
                    }
                    if len(results["failure_examples"]) < 5: results["failure_examples"].append(rec)
                    fail_rec.append(rec)
                    # Limit failure examples per constraint in log to 3
                    if len(fail_rec) >= 3: break
            
            # Log
            log_entry = {
                "id": cid, 
                "kind": constr.kind, 
                "tests_run": len(targets), 
                "failures_count": len(fail_rec),
                "failure_examples": fail_rec
            }
            state.fuzz_log.append(log_entry)

        return json.dumps(results)

    # Standard Actions
    def update_plan(self, state: AgentState, plan_data: Dict[str, Any]) -> str:
        # 1. Initialize empty plan if None
        if state.plan is None:
            state.plan = {
                "observations": [],
                "variables": [],
                "constraints": [],
                "strategy": "",
                "verification": "DRAFT",
                "current_code": [],
                "problems": []
            }

        # 2. Merge Updates
        # We allow partial updates. If a key is present in plan_data, it overwrites/appends.
        for key in ["observations", "variables", "constraints", "strategy", "verification", "current_code", "problems", "problem_notes"]:
            if key in plan_data:
                val = plan_data[key]
                # Sanitize Observations (Flatten Objects to Strings)
                if key == "observations" and isinstance(val, list):
                    clean_obs = []
                    for item in val:
                        if isinstance(item, dict):
                             # Flatten {"key": "k", "value": "v"} -> "k: v"
                             parts = [f"{k}: {v}" for k,v in item.items()]
                             clean_obs.append(", ".join(parts))
                        else:
                             clean_obs.append(str(item))
                    val = clean_obs
                
                state.plan[key] = val
                
                # Sanitize Variables (Flatten Lists/Objects to format suitable for Plan)
                if key == "variables" and isinstance(val, list):
                    clean_vars = []
                    for item in val:
                        if isinstance(item, list):
                             # Flatten nested ["R1", "x"] -> "R1_x" or join
                             clean_vars.append("_".join([str(sub) for sub in item]))
                        elif isinstance(item, dict):
                             # {"name": "x", "type": "bool"} -> "x"
                             clean_vars.append(item.get("name", str(item)))
                        else:
                             clean_vars.append(str(item))
                    state.plan[key] = clean_vars

        # 3. Check for completeness only on Confirmation
        if state.plan.get("verification") == "CONFIRMED":
            failures = []
            if not state.plan.get("observations"): failures.append("observations")
            if not state.plan.get("variables"): failures.append("variables")
            if not state.plan.get("strategy"): failures.append("strategy")
            if failures:
                 # Revert status to DRAFT if incomplete
                 state.plan["verification"] = "REFINING"
                 return f"Cannot CONFIRM plan yet. Missing meaningful content in: {failures}. Status reverted to REFINING."

        self._write_plan_to_file(state)
        
        # --- DETERMINISTIC PHASE GUIDANCE ---
        current = state.current_phase
        
        if current == AgentPhase.PLANNING:
            status = state.plan.get("verification", "DRAFT")
            if status == "CONFIRMED":
                 return "Plan updated and CONFIRMED. You MUST now call 'ADVANCE_PHASE' to proceed to IMPLEMENTATION."
            return f"Plan updated (Status: {status}). You must define Observations, Variables, and Constraints, then set verification to 'CONFIRMED'."

        return f"Plan updated (Status: {state.plan.get('verification')})."

    def finish_action(self, state: AgentState, args: Dict[str, Any]) -> str:
        state.finished = True
        state.final_status = "FINISHED"

        # Agent-Driven Reporting
        if "report" in args:
             try:
                 with open("output.txt", "w") as f:
                     f.write(args["report"])
                 return "Terminating. Final Report written to output.txt."
             except Exception as e:
                 return f"Terminating. Failed to write report: {e}"

        return "Terminating"

    def advance_phase(self, state: AgentState, _: Any = None) -> str:
        current = state.current_phase
        
        # Validation before leaving current phase
        if current == AgentPhase.PLANNING:
            if state.plan.get("verification") != "CONFIRMED":
                return "Error: Cannot advance from PLANNING. You must set 'verification': 'CONFIRMED' in your plan first."
            state.current_phase = AgentPhase.IMPLEMENTATION
            return "Phase advanced to IMPLEMENTATION. Goal: Write MiniZinc code using 'UPDATE_MODEL_FILE' and then 'SOLVE'."

        elif current == AgentPhase.IMPLEMENTATION:
            # If we are here, we might be stuck or failed solving. Move to DEBUGGING.
            state.current_phase = AgentPhase.DEBUGGING
            return "Phase advanced to DEBUGGING. Goal: Review solution, fix issues, or refine model."
            
        elif current == AgentPhase.DEBUGGING:
            # Loop back to IMPLEMENTATION to retry
            state.current_phase = AgentPhase.IMPLEMENTATION
            return "Phase cycled back to IMPLEMENTATION. Retry solving."

        return f"Already in final phase {current}"

    def _write_plan_to_file(self, state: AgentState):
        try:
            content = "# Implementation Plan\n\n"
            content += f"## Verification Status\n**{state.plan.get('verification', 'DRAFT')}**\n\n"
            content += f"## Observations\n{json.dumps(state.plan.get('observations', []), indent=2)}\n\n"
            content += f"## Variables\n{json.dumps(state.plan.get('variables', []), indent=2)}\n\n"
            content += f"## Constraints\n{json.dumps(state.plan.get('constraints', []), indent=2)}\n\n"
            content += f"## Strategy\n{state.plan.get('strategy', '')}\n\n"
            content += f"## Current Code\n```json\n{json.dumps(state.plan.get('current_code', []), indent=2)}\n```\n\n"
            content += f"## Problems\n{json.dumps(state.plan.get('problems', []), indent=2)}\n\n"
            
            # Generated Code Section
            try:
                backend = self.registry.get(state.active_ir_backend)
                code_view = backend.generate_code(state)
                content += f"## Generated Code (Backend: {backend.name})\n```text\n{code_view}\n```\n"
            except Exception as e:
                content += f"## Generated Code\nError generating code view: {e}\n"

            with open("PLAN.md", "w") as f:
                f.write(content)
            print(f"{DIM}SATManager: Wrote PLAN.md ({len(content)} bytes){RESET}")
        except Exception as e:
            print(f"{RED}SATManager: PLAN.md Write Failed: {e}{RESET}")
            return f"Plan updated in state, but failed to write PLAN.md: {e}"
        except Exception as e:
            return f"Plan updated in state, but failed to write PLAN.md: {e}"
        except Exception as e:
            return f"Plan updated in state, but failed to write PLAN.md: {e}"
            
        if state.plan['verification'] != "CONFIRMED":
            return "Plan updated (Status: " + state.plan['verification'] + "). You must REFINE and eventually set verification to 'CONFIRMED' to proceed."
        return "Plan verified and locked. You may now DEFINE_VARIABLES."

    def define_variables(self, state: AgentState, var_names: List[str]) -> str:
        if not state.plan: return "Error: You MUST call 'UPDATE_PLAN' before defining variables."
        if state.plan.get("verification") != "CONFIRMED": return "Error: Plan verification is NOT 'CONFIRMED'. You must iterate on the plan (checking bounds, symmetry, etc) and explicitly set verification to 'CONFIRMED' only when fully satisfied."
        added = []
        for name in var_names:
            if name not in state.sat_variables:
                state.sat_variables[name] = state.next_var_id
                state.next_var_id += 1
                added.append(name)
        
        if not added:
            return "No new variables registered. You have likely defined them all. Next Goal: Use 'ADD_MODEL_CONSTRAINTS' to forbid collisions or set assignments."
            
        return f"Registered {len(added)} variables: {added}"

    def add_model_constraints(self, state: AgentState, constraints_data: List[Dict[str, Any]]) -> str:
        if not state.plan: return "Error: You MUST call 'CREATE_PLAN' before adding constraints."
        
        # File-Centric Guard
        if state.model_file_path:
             return f"Error: You are in File-Centric mode ({state.model_file_path}). You cannot use 'ADD_MODEL_CONSTRAINTS'. Please use 'UPDATE_MODEL_FILE' to append constraints to the file."

        added_count = 0
        try:
            backend = self.registry.get(state.active_ir_backend)
            allowed = backend.allowed_kinds()
            for c_data in constraints_data:
                if "kind" not in c_data or "parameters" not in c_data:
                    return f"Error: Missing kind/parameters"
                kind = c_data["kind"]
                if kind not in allowed:
                    return f"Error: Kind '{kind}' not allowed in {backend.name}"
                
                # Strict Backend Enforcement
                if "ir_backend" in c_data and c_data["ir_backend"] != state.active_ir_backend:
                     return f"Error: Cannot add constraint for backend '{c_data['ir_backend']}' while active backend is '{state.active_ir_backend}'"

                # Category Inference
                cat = c_data.get("category")
                if not cat:
                    if kind in ["at_most_k", "at_least_k", "exactly_k", "exactly_one", "alldifferent_onehot"]: cat = "cardinality"
                    elif kind in ["linear_leq", "linear_eq"]: cat = "geometry"
                    elif kind in ["clause", "implies"]: cat = "logic"
                    elif kind == "connected_8": cat = "connectivity"
                    else: cat = "logic"
                
                c_id = c_data.get("id", str(uuid.uuid4()))
                c = ModelingConstraint(id=c_id, ir_backend=backend.name, kind=kind, parameters=c_data["parameters"], category=cat)
                backend.validate_constraint(c, state)
                state.model_constraints.append(c)
                added_count += 1
        except Exception as e:
            return f"Error: {str(e)}"
        return f"Added {added_count} constraints."

    def add_minizinc_code(self, state: AgentState, code: str) -> str:
        if not state.plan: return "Error: You MUST call 'UPDATE_PLAN' before adding code."
        if "backend" in self.registry.get(state.active_ir_backend).name and "minizinc" not in self.registry.get(state.active_ir_backend).name:
             return f"Error: Active backend is {state.active_ir_backend}, raw MiniZinc code is likely invalid."
        
        # Basic validation (non-empty)
        if not code.strip(): return "Error: Empty code block."
        
        state.minizinc_code.append(code)
        return "Added raw MiniZinc code block."

    def update_model_file(self, state: AgentState, args: Dict[str, Any]) -> str:
        if not state.plan: return "Error: Call UPDATE_PLAN first."
        if state.current_phase == AgentPhase.PLANNING:
             return f"Error: You are in PLANNING phase. You cannot write code files yet. Please use 'UPDATE_PLAN' to conceptualize your strategy. Confirm plan to advance to IMPLEMENTATION."
        
        content = args.get("content", "")
        mode = args.get("mode", "append") # 'append' or 'overwrite'
        
        # Determine path
        if not state.model_file_path:
            # Create a unique path or standard path
            # We use a standard path per run to allow clean overrides
            state.model_file_path = "memory/current_model.mzn"
            
        # Ensure dir exists
        import os
        os.makedirs(os.path.dirname(state.model_file_path), exist_ok=True)
        
        try:
            file_mode = "a" if mode == "append" else "w"
            # If append and file doesn't exist, it creates it.
            # If overwrite, it truncates.
            
            with open(state.model_file_path, file_mode) as f:
                if mode == "append" and os.path.getsize(state.model_file_path) > 0:
                    f.write("\n") # Ensure newline separator
                f.write(content)
            
            size = os.path.getsize(state.model_file_path)
            print(f"{DIM}SATManager: Updated file {state.model_file_path} ({size} bytes){RESET}")
            return f"Success. Updated {state.model_file_path} (Mode: {mode}). New Size: {size} bytes."
        except Exception as e:
            return f"File I/O Error: {e}"

    def read_model_file(self, state: AgentState) -> str:
        if not state.model_file_path: return "No active model file created yet."
        import os
        if not os.path.exists(state.model_file_path): return "File does not exist."
        
        try:
            with open(state.model_file_path, "r") as f:
                content = f.read()
            return f"--- FILE: {state.model_file_path} ---\n{content}\n--- END FILE ---"
        except Exception as e:
            return f"Read Error: {e}"

    def remove_model_constraints(self, state: AgentState, ids: List[str]) -> str:
        # File-Centric Guard
        if state.model_file_path:
             return f"Error: You are in File-Centric mode ({state.model_file_path}). You cannot use 'REMOVE_MODEL_CONSTRAINTS'. The constraints are in the file, not in the object list. Please use 'UPDATE_MODEL_FILE(mode='overwrite', content=...)' to rewrite the file with the fix."

        orig = len(state.model_constraints)
        state.model_constraints = [c for c in state.model_constraints if c.id not in ids]
        return f"Removed {orig - len(state.model_constraints)} constraints."

    def get_schema(self, state: AgentState) -> str:
        backend = self.registry.get(state.active_ir_backend)
        return json.dumps(backend.allowed_kinds(), indent=2)

    def test_constraint(self, state: AgentState, args: Dict[str, Any]) -> str:
        """Isolated check: can we compile these specific constraints?"""
        c_payloads = args.get("constraints", [])
        if not c_payloads: return "No constraints provided to test."
        
        # Temp build
        test_constraints = []
        backend = self.registry.get(state.active_ir_backend)
        
        try:
            for c_data in c_payloads:
                kind = c_data.get("kind")
                if kind not in backend.allowed_kinds(): return f"Invalid Kind: {kind}"
                c = ModelingConstraint(
                    id="test", ir_backend=backend.name, 
                    kind=kind, parameters=c_data.get("parameters", {})
                )
                backend.validate_constraint(c, state)
                test_constraints.append(c)
            
            # Isolated Compile
            clauses = self.compile_subset(state, test_constraints)
            return f"Valid. Compiled to {len(clauses)} clauses."
        except Exception as e:
            return f"Invalid: {str(e)}"

    def add_constraints(self, state: AgentState, constraints: List[List[str]]) -> str:
        return "Error: Raw clause injection is disabled. Use ADD_MODEL_CONSTRAINTS."

    def refine_from_validation(self, state: AgentState, errors: List[str]) -> str:
        # This action is a signal for the LLM to start fixing things.
        # It doesn't change state directly but logs the intent to refine.
        return f"Refining model based on {len(errors)} validation errors. Please REMOVE or ADD constraints."

    def solve(self, state: AgentState) -> str:
        # Hybrid Solve Logic: Delegate to Backend if it supports solving
        backend = self.registry.get(state.active_ir_backend)
        if hasattr(backend, "solve") and callable(backend.solve):
            try:
                # The backend handles specific compilation and solving
                msg = backend.solve(state)
                # Ensure generic success message for agent flow if strictly valid
                if "Solution Found" in msg:
                    # Depending on backend, solution might be stored in state.solution
                    return "Solution Found (Stored in state). Validation Pending."
                return msg
            except Exception as e:
                return f"Backend Solve Error: {e}"

        if hasattr(backend, "solve") and callable(backend.solve):
            try:
                # The backend handles specific compilation and solving
                print(f"{DIM}SATManager: Delegating solve to backend {backend.name}...{RESET}")
                msg = backend.solve(state)
                # Ensure generic success message for agent flow if strictly valid
                if "Solution Found" in msg:
                    # Depending on backend, solution might be stored in state.solution
                    print(f"{GREEN}SATManager: Backend returned Solution Found.{RESET}")
                    return "Solution Found (Stored in state). Validation Pending."
                print(f"{RED}SATManager: Backend msg: {msg}{RESET}")
                return msg
            except Exception as e:
                return f"Backend Solve Error: {e}"

        # Default PySAT Logic (for 'pb', 'cnf')
        try:
            print(f"{DIM}SATManager: Compiling to CNF...{RESET}")
            self._compile(state)
        except Exception as e:
            return f"Compilation Failed: {e}"
        
        if not state.cnf_clauses: return "No Constraints compiled."

        try:
            print(f"{DIM}SATManager: Solving CNF ({len(state.cnf_clauses)} clauses)...{RESET}")
            s = Solver(name=self.solver_name, bootstrap_with=state.cnf_clauses)
            if s.solve():
                print(f"{GREEN}SATManager: SAT! Decoding model...{RESET}")
                m = s.get_model()
                assignment = {v: (v > 0) for v in m}
                result = {k: assignment[v] for k, v in state.sat_variables.items() if v in assignment}
                state.solution = result
                s.delete()
                # Important: Return generic message. Validation happens in Agent Loop.
                return "Solution Found (Stored in state). Validation Pending."
            s.delete()
            return "Unexpected UNSAT"
        except Exception as e: return f"Error: {e}"

    def decode_solution(self, state: AgentState) -> str:
        # Legacy stub
        if state.solution: return "Solution available in state."
        return "No solution."

    def refine_model(self, state: AgentState, feedback: str) -> str:
        return "Model Refinement Logged"
