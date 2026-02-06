import copy
import json
import uuid
import random
from typing import List, Dict, Any, Optional

from engine.state import AgentState, ModelingConstraint, AgentPhase
from engine.backends.registry import IRBackendRegistry
from engine.compilation.artifact import CompilationArtifact
from engine.vars import VarManager

# PySAT imports
try:
    from pysat.solvers import Solver
except ImportError:
    Solver = Any

from engine.debug_harness import DebugHarness
from engine.solution.types import SatResult, SatStatus, DomainSolution
import re

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
        # Deep copy vars to prevent pollution
        temp_state.var_manager = copy.deepcopy(state.var_manager)
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
        num_tests = payload.get("num_tests", 50)
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
            if targets:
                print(f"fuzzing using {len(targets)} generated test cases:")
                solver = Solver(name=self.solver_name, bootstrap_with=cnf)
                try:
                    for assump, expected, meta in targets:
                        # Map to IDs
                        ids = []
                        for a in assump:
                            is_neg = str(a).startswith('-') or str(a).startswith('~')
                            name = str(a).lstrip('-~')
                            var_id = state.sat_variables.get(name)
                            if var_id:
                                ids.append(-var_id if is_neg else var_id)
                        
                        status = solver.solve(assumptions=ids)
                        outcome = "SAT" if status else "UNSAT"
                        
                        if outcome == expected:
                            print(".", end="", flush=True)
                        else:
                            print("F", end="", flush=True)
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
                            if len(results["failure_examples"]) < 10: 
                                results["failure_examples"].append(rec)
                            fail_rec.append(rec)
                finally:
                    solver.delete()
                print("") # Close line
            
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
            if "minizinc" in state.active_ir_backend.lower():
                return "Phase advanced to IMPLEMENTATION. Goal: Write MiniZinc code using 'UPDATE_MODEL_FILE' and then 'SOLVE'."
            else:
                return "Phase advanced to IMPLEMENTATION. Goal: Use 'DEFINE_VARIABLE_PATTERN' and 'ADD_PYTHON_CONSTRAINT_BLOCK' to build model, then 'SOLVE'."

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
                # Truncate if too long (e.g. > 100 lines)
                lines = code_view.splitlines()
                if len(lines) > 110:
                    code_view = "\n".join(lines[:100]) + f"\n... [{len(lines)-100} more lines truncated] ..."
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
                state.var_manager.declare(name)
                added.append(name)
        
        if not added:
            return "No new variables registered. You have likely defined them all. Next Goal: Use 'ADD_MODEL_CONSTRAINTS' to forbid collisions or set assignments."
            
        return f"Registered {len(added)} variables: {added}"

    def define_variable_pattern(self, state: AgentState, args: Dict[str, Any]) -> str:
        """
        Generates variables based on a pattern and ranges.
        Args:
            pattern: f-string style pattern, e.g. "pos_{r}_{t}"
            ranges: Dict of range config, e.g. {"r": 5, "t": [0, 1, 2]} or {"r": 5} (implies 0..4)
        """
        if not state.plan: return "Error: You MUST call 'UPDATE_PLAN' before defining variables."
        
        pattern = args.get("pattern")
        ranges_cfg = args.get("ranges")
        if not pattern or not ranges_cfg:
             return "Error: Must provide 'pattern' (str) and 'ranges' (dict)."
        
        import itertools
        
        keys = sorted(ranges_cfg.keys())
        iterables = []
        for k in keys:
             val = ranges_cfg[k]
             if isinstance(val, int):
                 iterables.append(range(val))
             elif isinstance(val, list):
                 iterables.append(val)
             else:
                 return f"Error: Range for '{k}' must be int (count) or list (explicit values)."
        
        added = []
        # Cartesian Product
        for values in itertools.product(*iterables):
             ctx = dict(zip(keys, values))
             try:
                 name = pattern.format(**ctx)
             except KeyError as e:
                 return f"Error: Pattern key {e} not found in ranges."
             
             if name not in state.sat_variables:
                 state.sat_variables[name] = state.next_var_id
                 state.next_var_id += 1
                 state.var_manager.declare(name)
                 added.append(name)
        
        return f"Generated {len(added)} variables from pattern '{pattern}'. Examples: {added[:5]}..."

    
    def add_python_constraint_block(self, state: AgentState, code: str) -> str:
        """
        Executes a block of Python code to generate constraints.
        Exposes:
        - clause(lits) / add_clause(lits)
        - at_most_k(vars, k) / add_at_most_k(vars, k)
        - exactly_one(vars) / add_exactly_one(vars)
        - implies(a, b) / add_implies(a, b)
        - linear_leq(terms, rhs) / add_linear_leq(terms, rhs)
        - variables: dict (read-only copy of state.sat_variables)
        """
        if not state.plan: return "Error: Plan required."
        
        # Sync legacy sat_variables to var_manager just in case
        for k in state.sat_variables:
            state.var_manager.declare(k)
        
        # 1. Setup Scope
        added_count = 0
        errors = []
        
        def _add(kind, params):
            nonlocal added_count
            # Validate vars exist
            for key, val in params.items():
                if key == "vars" or key == "literals":
                    for v in val:
                        s_v = str(v).lstrip("-~")
                        if s_v not in state.sat_variables:
                            errors.append(f"Variable '{s_v}' not defined.")
                            return
                if key == "terms":
                     for t in val:
                         s_v = str(t['var']).lstrip("-~")
                         if s_v not in state.sat_variables:
                             errors.append(f"Variable '{s_v}' not defined.")
                             return

            c_id = f"c_{uuid.uuid4().hex[:8]}"
            c = ModelingConstraint(id=c_id, kind=kind, parameters=params, ir_backend=state.active_ir_backend)
            state.model_constraints.append(c)
            added_count += 1
            return c

        # Helper Functions exposed to the Agent
        def clause(lits): return _add("clause", {"literals": lits})
        def at_most_k(vars, k): return _add("at_most_k", {"vars": vars, "k": k})
        def exactly_one(vars): return _add("exactly_one", {"vars": vars})
        def exactly_k(vars, k): return _add("exactly_k", {"vars": vars, "k": k})
        def implies(a, b): return _add("implies", {"a": a, "b": b})
        def linear_leq(terms, rhs): return _add("linear_leq", {"terms": terms, "rhs": rhs})
        def linear_eq(terms, rhs): return _add("linear_eq", {"terms": terms, "rhs": rhs})
        
        # Context
        scope = {
            "clause": clause, "add_clause": clause,
            "at_most_k": at_most_k, "add_at_most_k": at_most_k,
            "exactly_one": exactly_one, "add_exactly_one": exactly_one,
            "exactly_k": exactly_k, "add_exactly_k": exactly_k,
            "implies": implies, "add_implies": implies,
            "linear_leq": linear_leq, "add_linear_leq": linear_leq,
            "linear_eq": linear_eq, "add_linear_eq": linear_eq,
            "variables": state.sat_variables.copy(),
            "range": range,
            "list": list,
            "dict": dict,
            "print": print,
            # Robustness: ignore accidental tool-calls inside the block
            "ADD_MODEL_CONSTRAINTS": lambda x: None,
            "ADD_PYTHON_CONSTRAINT_BLOCK": lambda x: None,
            "add_constraints": lambda x: [_add(c["kind"], c["parameters"]) if isinstance(c, dict) else None for c in (x if isinstance(x, list) else [x])],
            "ADD_CONSTRAINTS": lambda x: [_add(c["kind"], c["parameters"]) if isinstance(c, dict) else None for c in (x if isinstance(x, list) else [x])]
        }
        
        # 2. Execute
        try:
            exec(code, scope)
        except Exception as e:
            return f"Error executing constraint block: {e}"
            
        if errors:
            return f"Execution run but encountered {len(errors)} variable errors. First error: {errors[0]}"
            
        return f"Successfully executed block. Added {added_count} constraints."

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
            if state.metrics: state.metrics["syntax_errors"] += 1
            print(f"{RED}SATManager: Model Rejected (File I/O Error): {e}{RESET}")
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
        import time
        start_time = time.time()
        
        # Denabase: Initialize Trace
        try:
            from Denabase.Denabase.trace import EncodingTrace
            state.denabase_trace = EncodingTrace(summary={"backend": state.active_ir_backend})
        except ImportError:
            pass # Denabase not installed
        
        print(f"{DIM}SATManager: Solving with Backend ({state.active_ir_backend})...{RESET}")
        
        result = SatResult(status=SatStatus.UNKNOWN)
        
        try:
            # 1. Sync Legacy variables (safety net)
            for name in state.sat_variables:
                state.var_manager.declare(name)

            # 2. Compile ONCE
            backend = self.registry.get(state.active_ir_backend)
            print(f"{DIM}SATManager: Compiling to Artifact...{RESET}")
            
            artifact = backend.compile(state)
            state.compilation_artifact = artifact
            state.cnf_clauses = artifact.clauses # Legacy compat
            
            # 3. Setup Harness
            # Safety Filter: PySAT crashes on 0 literals. Preserve indices for provenance.
            safe_clauses = []
            for c_idx, c in enumerate(artifact.clauses):
                if 0 in c:
                    # Find which constraint this belongs to
                    source_cid = "Unknown"
                    for cid, indices in artifact.constraint_to_clause_ids.items():
                         if c_idx in indices:
                             source_cid = cid
                             break
                    print(f"{YELLOW}Warning: Clause {c_idx} (from {source_cid}) contains literal 0, pruning: {c}{RESET}")
                    cleaned = [l for l in c if l != 0]
                    if not cleaned:
                         print(f"{RED}Error: Clause {c_idx} became EMPTY after pruning 0 literals!{RESET}")
                    safe_clauses.append(cleaned)
                else:
                    safe_clauses.append(c)

            print(f"{DIM}SATManager: Initializing DebugHarness...{RESET}")
            harness = DebugHarness(solver_name=self.solver_name)
            # We use reset_instrumentation instead of load_problem if we intend to use groups
            harness.reset_instrumentation(state.var_manager.max_id)
            
            # Register groups from Artifact
            if artifact.constraint_to_clause_ids:
                for cid, indices in artifact.constraint_to_clause_ids.items():
                    if not indices: continue
                    # These indices refer to the original artifact.clauses, so we need to map them
                    # to the safe_clauses if any were filtered out.
                    # For now, we assume the indices are still valid for the filtered list,
                    # which implies that if a clause was filtered, its index is skipped.
                    # A more robust solution would re-index, but this is simpler for now.
                    # If a clause with 0 was filtered, its original index might not exist in safe_clauses.
                    # Let's adjust to use the original clauses for group mapping, and the filtered for solving.
                    # This means the group mapping might contain clauses that were filtered out.
                    # A better approach would be to re-map indices after filtering.
                    # For now, we'll pass the original clauses to add_group, and safe_clauses to the solver.
                    c_clauses = [safe_clauses[i] for i in indices]
                    harness.add_group(cid, c_clauses, {"id": cid})
            else:
                 harness.add_group("root", safe_clauses, {"id": "root"})
            
            # 4. Solve (Enable all groups)
            all_sels = [g["selector"] for g in harness.groups.values()]
            print(f"{DIM}SATManager: Solving with {len(all_sels)} enabled groups...{RESET}")
            is_sat, model = harness.solve(assumptions=all_sels)
            
            result.time_taken = (time.time() - start_time) * 1000
            
            if is_sat:
                print(f"{GREEN}SATManager: SAT! Decoding... (Time: {result.time_taken:.1f}ms){RESET}")
                
                result.status = SatStatus.SAT
                assignment = {v: (v > 0) for v in model}
                
                # Decode Model Names
                # Use artifact.var_map names
                decoded_model = {}
                for name, vid in artifact.var_map.items():
                    if "::" in name: continue 
                    # Handle boolean logic
                    if vid in assignment:
                        decoded_model[name] = assignment[vid]
                    elif -vid in assignment:
                        decoded_model[name] = assignment[-vid]
                    else:
                        decoded_model[name] = False
                
                result.model = decoded_model
                state.sat_result = result
                
                # Auto-Decode Domain Solution
                self.decode_solution(state)
                
                if "metrics" in state.serialize(): state.metrics["iterations_to_valid"] += 1
                
                # Denabase: Store Success
                if state.denabase_trace:
                    try:
                        from Denabase.agent.denabase_bridge import DenabaseBridge
                        bridge = DenabaseBridge.get_instance()
                        # Create generic ID
                        pid = f"auto_{uuid.uuid4().hex[:6]}"
                        eid = bridge.create_solution_entry(
                            family="agent_auto", 
                            problem_id=pid, 
                            cnf_clauses=safe_clauses,
                            meta={"source": "agent", "backend": state.active_ir_backend}
                        )
                        bridge.attach_trace(eid, state.denabase_trace)
                        print(f"{GREEN}Denabase: Stored solution trace as {pid} ({eid}){RESET}")
                    except Exception as e:
                        print(f"{YELLOW}Denabase Store Failed: {e}{RESET}")

                return f"Solution Found (SAT). Time: {result.time_taken:.1f}ms. Domain decoding applied."
            
            else:
                print(f"{YELLOW}SATManager: UNSAT. Diagnosing...{RESET}")
                result.status = SatStatus.UNSAT
                if "metrics" in state.serialize(): 
                    state.metrics["rejections"] += 1
                    state.metrics["unsat_cores_trigger"] += 1
                
                report = harness.diagnose()
                
                # Store conflict info in result
                result.unsat_core = report.get("minimized_core", [])
                result.conflict_info = report.get("conflict_info", {})
                
                state.sat_result = result
                
                # Simplify report for Agent
                msg = "Constraint Satisfaction Failed (UNSAT).\n"
                if result.unsat_core:
                    msg += "Conflict detected between:\n"
                    for g in result.unsat_core[:5]:
                        info = result.conflict_info.get(g, {})
                        kind = info.get("kind", "unknown")
                        label = f"Group '{g}' ({kind})"
                        if "raw" in info:
                             label += f": {info['raw'][:50]}..."
                        msg += f"- {label}\n"
                    if len(result.unsat_core) > 5:
                        msg += f"... and {len(result.unsat_core)-5} more."
                return msg

        except Exception as e:
            if "metrics" in state.serialize(): state.metrics["rejections"] += 1
            import traceback
            traceback.print_exc()
            return f"Pipeline Error: {e}"

    def decode_solution(self, state: AgentState) -> str:
        if not state.sat_result: return "No execution result available."
        
        res = state.sat_result
        if res.status != SatStatus.SAT or not res.model:
            state.domain_solution = DomainSolution(is_valid=False)
            return "Result is Not SAT."
            
        print(f"{DIM}SATManager: MRPP/Domain Decoding...{RESET}")
        
        # Domain Decoding
        ds = DomainSolution(is_valid=True, variables=res.model, raw_result=res)
        
        # MRPP Heuristic
        try:
            paths = self._decode_mrpp(state, res.model)
            if paths:
                ds.paths = paths
                print(f"{GREEN}SATManager: Decoded paths for {len(paths)} agents.{RESET}")
        except Exception as e:
            print(f"{YELLOW}SATManager: MRPP Decode Warning: {e}{RESET}")
            
        state.domain_solution = ds
        # Prefer structured domain solution for the checker, fallback to raw model
        state.solution = {"paths": ds.paths} if ds.paths else res.model
        
        summary = f"Solution Valid. Variables: {len(res.model)}.\n"
        if ds.paths:
             summary += "Decoded Paths:\n"
             for r, p in ds.paths.items():
                 summary += f"  Agent {r}: {p}\n"
        return summary

    def _decode_mrpp(self, state: AgentState, model: Dict[str, bool]) -> Dict[str, List[Any]]:
        """
        Heuristic decoder for patterns:
        1. pos_{r}_{x}_{y}_{t} (Boolean Grid)
        2. pos_{r}_{loc}_{t} (Graph Node)
        3. robot_x[r,t] / robot_y[r,t] (Integer/MiniZinc)
        """
        # 1. Regex patterns for different representations
        # pos[r][x][y][t] or pos_r_x_y_t
        p_r_x_y_t = re.compile(r"pos[._]([a-zA-Z0-9]+)[._](\d+)[._](\d+)[._](\d+)")
        # pos[r][t][x][y] (Common in MiniZinc)
        p_r_t_x_y = re.compile(r"pos[._]([a-zA-Z0-9]+)[._]([a-zA-Z0-9]+)[._](\d+)[._](\d+)")
        # pos[r][loc][t]
        p_node = re.compile(r"pos[._]([a-zA-Z0-9]+)[._]([a-zA-Z0-9\_]+)[._](\d+)")
        
        # Store for heuristic discovery
        # Map: robot -> list of (t, x, y) or (t, loc)
        raw_paths = {} 
        
        grid_count = 0
        node_count = 0
        true_vars = [k for k, v in model.items() if v]

        for var in true_vars:
            # Try r, x, y, t
            m = p_r_x_y_t.search(var) 
            if m:
                r, x, y, t = m.groups()
                # Heuristic check: is it really r,x,y,t or r,t,x,y?
                # If t > 100 and x < 10, it's probably r,t,x,y handled by the second regex
                if r not in raw_paths: raw_paths[r] = []
                raw_paths[r].append((int(t), [int(x), int(y)]))
                grid_count += 1
                continue
            
            # Try r, t, x, y
            m = p_r_t_x_y.search(var)
            if m:
                r, t, x, y = m.groups()
                if r not in raw_paths: raw_paths[r] = []
                raw_paths[r].append((int(t), [int(x), int(y)]))
                grid_count += 1
                continue
                
            # Check Node
            m = p_node.search(var)
            if m:
                r, loc, t = m.groups()
                try:
                    t_val = int(t)
                    if r not in raw_paths: raw_paths[r] = []
                    raw_paths[r].append((t_val, loc))
                    node_count += 1
                except: continue

        # 2. MiniZinc specialized check (robot_x[r,t], robot_y[r,t])
        # Note: model might contain "robot_x[0,0] = 1" or similar
        # We also look for robot_x_0_0 and robot_x[0,0]
        p_mzn_x = re.compile(r"robot[._]x[._\[](\d+)[,._](\d+)")
        p_mzn_y = re.compile(r"robot[._]y[._\[](\d+)[,._](\d+)")
        
        mzn_x = {} # (r, t) -> int
        mzn_y = {} # (r, t) -> int
        
        for var, val in model.items():
            mx = p_mzn_x.search(var)
            if mx:
                 r_idx, t_idx = map(int, mx.groups())
                 mzn_x[(r_idx, t_idx)] = int(val)
                 continue
            my = p_mzn_y.search(var)
            if my:
                 r_idx, t_idx = map(int, my.groups())
                 mzn_y[(r_idx, t_idx)] = int(val)

        if mzn_x and mzn_y:
            for (r, t), x_val in mzn_x.items():
                if (r, t) in mzn_y:
                    r_str = str(r)
                    if r_str not in raw_paths: raw_paths[r_str] = []
                    raw_paths[r_str].append((t, [x_val, mzn_y[(r, t)]]))
                    grid_count += 1
        
        if grid_count or node_count:
             print(f"{DIM}SATManager: Decoded {grid_count} grid points and {node_count} node points.{RESET}")
        else:
             print(f"{YELLOW}SATManager: No MRPP patterns matched in model.{RESET}")
             if true_vars:
                  print(f"{YELLOW}Sample True Variables: {true_vars[:10]}{RESET}")
                  if any("X_INTRODUCED" in v for v in true_vars[:100]):
                       print(f"{YELLOW}SATManager: Detected 'X_INTRODUCED' variables. This usually means names were lost in translation (e.g. MiniZinc bit-blasting).{RESET}")
            
        # 3. Sort and Format
        final_paths = {}
        
        # Heuristic: try to map numerical indices to names if found in state.plan/observations
        id_to_name = {}
        if state.plan and "observations" in state.plan:
             obs_str = str(state.plan["observations"])
             m_names = sorted(list(set(re.findall(r"\b([A-Z])\b", obs_str))))
             if len(m_names) == len(raw_paths):
                  for i, name in enumerate(m_names):
                       id_to_name[str(i)] = name
             elif len(m_names) > len(raw_paths):
                  # Heuristic: map based on first mention? 
                  # For now just use numerical ids if ambiguous
                  pass

        for r, bumps in raw_paths.items():
            # Sort by T
            bumps.sort(key=lambda x: x[0])
            
            # The checker expects a list where index is T
            # [[x1, y1], [x2, y2], ...]
            # Assume continuous for now
            if not bumps: continue
            max_t = max(b[0] for b in bumps)
            path_list = [None] * (max_t + 1)
            for t, pos in bumps:
                if t <= max_t:
                    path_list[t] = pos
            
            # Use name if mapped, else stringified ID
            # Handle R0/Robot0 prefixes
            clean_r = re.sub(r'^[Rr]obot', '', str(r)).lstrip('0')
            if not clean_r: clean_r = "0" # Special case for Robot0
            
            name_key = id_to_name.get(clean_r, str(r))
            final_paths[name_key] = path_list
            
        return final_paths

    def refine_model(self, state: AgentState, feedback: str) -> str:
        return "Model Refinement Logged"
