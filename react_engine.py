import json
import dataclasses
from enum import Enum
import os
import abc
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
import functools
import random
import copy

# Abstracting PySAT imports
try:
    from pysat.solvers import Solver
    from pysat.formula import CNF
    from pysat.card import CardEnc
    from pysat.pb import PBEnc
except ImportError:
    Solver = Any
    CNF = Any
    CardEnc = Any
    PBEnc = Any

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclasses.dataclass
class IRConfig:
    backend: str
    backend_params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_env_or_file() -> 'IRConfig':
        # 1. Try Env Var
        env_backend = os.environ.get("IR_BACKEND")
        if env_backend:
            return IRConfig(backend=env_backend)
        
        # 2. Try Config Path
        config_path = os.environ.get("IR_CONFIG_PATH")
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    return IRConfig(
                        backend=data.get("backend", "cnf"),
                        backend_params=data.get("backend_params", {})
                    )
            except Exception:
                pass
        
        # Default
        return IRConfig(backend="cnf")


# ==============================================================================
# STATE & DATA STRUCTURES
# ==============================================================================

@dataclasses.dataclass
class ModelingConstraint:
    id: str
    ir_backend: str 
    kind: str       
    parameters: Dict[str, Any]
    source_text: Optional[str] = None

    def serialize(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

@dataclasses.dataclass
class AgentState:
    trajectory: List[Tuple[str, str, str]] = dataclasses.field(default_factory=list)
    sat_variables: Dict[str, int] = dataclasses.field(default_factory=dict)
    next_var_id: int = 1
    model_constraints: List[ModelingConstraint] = dataclasses.field(default_factory=list)
    active_ir_backend: str = "cnf"
    cnf_clauses: List[List[int]] = dataclasses.field(default_factory=list)
    step_count: int = 0
    finished: bool = False
    final_status: Optional[str] = None
    solution: Optional[Dict[str, Any]] = None
    fuzz_log: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def serialize(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ==============================================================================
# MODULAR IR BACKEND SYSTEM
# ==============================================================================

class IRBackend(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def validate_constraint(self, constraint: ModelingConstraint, state: AgentState) -> None:
        pass

    @abc.abstractmethod
    def compile_constraints(self, constraints: List[ModelingConstraint], state: AgentState) -> List[List[int]]:
        pass

    @abc.abstractmethod
    def allowed_kinds(self) -> Dict[str, Any]:
        pass


class CNFBackend(IRBackend):
    @property
    def name(self) -> str:
        return "cnf"

    def allowed_kinds(self) -> Dict[str, Any]:
        return {
            "clause": {
                "parameters": {"literals": "List[str]"},
                "description": "Standard disjunction of literals."
            }
        }

    def validate_constraint(self, constraint: ModelingConstraint, state: AgentState) -> None:
        if constraint.kind != "clause":
            raise ValueError(f"CNFBackend only supports kind='clause', got '{constraint.kind}'")
        lits = constraint.parameters.get("literals")
        if not isinstance(lits, list) or not lits:
            raise ValueError("CNF clause must have non-empty 'literals' list parameter.")
        for lit in lits:
            atom = str(lit).lstrip('-~')
            if atom not in state.sat_variables:
                raise ValueError(f"Variable '{atom}' not defined.")

    def compile_constraints(self, constraints: List[ModelingConstraint], state: AgentState) -> List[List[int]]:
        output_clauses = []
        for constr in constraints:
            if constr.ir_backend == self.name and constr.kind == "clause":
                clause_ints = []
                for lit in constr.parameters["literals"]:
                    is_neg = str(lit).startswith('-') or str(lit).startswith('~')
                    atom = str(lit).lstrip('-~')
                    if atom in state.sat_variables:
                        var_id = state.sat_variables[atom]
                        clause_ints.append(-var_id if is_neg else var_id)
                if clause_ints:
                    output_clauses.append(clause_ints)
        return output_clauses


class PBBackend(IRBackend):
    @property
    def name(self) -> str:
        return "pb"

    def allowed_kinds(self) -> Dict[str, Any]:
        return {
            "implies": {"parameters": {"a": "str", "b": "str"}},
            "at_most_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "at_least_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "exactly_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "exactly_one": {"parameters": {"vars": "List[str]"}},
            "linear_leq": {"parameters": {"terms": "List[Dict{var, coef}]", "rhs": "int"}},
            "linear_eq": {"parameters": {"terms": "List[Dict{var, coef}]", "rhs": "int"}}
        }

    def validate_constraint(self, constraint: ModelingConstraint, state: AgentState) -> None:
        kind = constraint.kind
        params = constraint.parameters
        if kind == "implies":
            if "a" not in params or "b" not in params: raise ValueError("implies requires 'a' and 'b'")
            for k in ["a", "b"]:
                if str(params[k]).lstrip('-~') not in state.sat_variables:
                    raise ValueError(f"Var {params[k]} not found")
        elif kind in ["at_most_k", "at_least_k", "exactly_k"]:
            if "vars" not in params or "k" not in params: raise ValueError(f"{kind} requires 'vars' and 'k'")
            if not isinstance(params["vars"], list): raise ValueError("'vars' must be list")
            for v in params["vars"]:
                if str(v).lstrip('-~') not in state.sat_variables: raise ValueError(f"Var {v} not found")
        elif kind == "exactly_one":
            if "vars" not in params or not isinstance(params["vars"], list): raise ValueError("'vars' must be list")
            for v in params["vars"]:
                if str(v).lstrip('-~') not in state.sat_variables: raise ValueError(f"Var {v} not found")
        elif kind in ["linear_leq", "linear_eq"]:
            if "terms" not in params or "rhs" not in params: raise ValueError(f"{kind} requires 'terms' and 'rhs'")
            for t in params["terms"]:
                if "var" not in t or "coef" not in t: raise ValueError("Term must have 'var' and 'coef'")
                if str(t["var"]).lstrip('-~') not in state.sat_variables: raise ValueError(f"Var {t['var']} not found")
        else:
            raise ValueError(f"Unknown PB kind: {kind}")

    def compile_constraints(self, constraints: List[ModelingConstraint], state: AgentState) -> List[List[int]]:
        output_clauses = []
        top_id = state.next_var_id - 1
        
        def get_lit(name):
            name_str = str(name)
            is_neg = name_str.startswith('-') or name_str.startswith('~')
            atom = name_str.lstrip('-~')
            vid = state.sat_variables.get(atom)
            if vid is None: raise ValueError(f"Missing var {atom}")
            return -vid if is_neg else vid

        for constr in constraints:
            if constr.ir_backend != self.name: continue
            
            cnf_obj = None
            k = constr.kind
            p = constr.parameters
            
            if k == "implies":
                output_clauses.append([-get_lit(p["a"]), get_lit(p["b"])])
            elif k == "at_most_k":
                cnf_obj = CardEnc.atmost([get_lit(v) for v in p["vars"]], int(p["k"]), top_id=top_id)
            elif k == "at_least_k":
                cnf_obj = CardEnc.atleast([get_lit(v) for v in p["vars"]], int(p["k"]), top_id=top_id)
            elif k == "exactly_k":
                cnf_obj = CardEnc.equals([get_lit(v) for v in p["vars"]], int(p["k"]), top_id=top_id)
            elif k == "exactly_one":
                cnf_obj = CardEnc.equals([get_lit(v) for v in p["vars"]], 1, top_id=top_id)
            elif k in ["linear_leq", "linear_eq"]:
                lits = [get_lit(t["var"]) for t in p["terms"]]
                weights = [int(t["coef"]) for t in p["terms"]]
                rhs = int(p["rhs"])
                if k == "linear_leq":
                    cnf_obj = PBEnc.leq(lits, weights, rhs, top_id=top_id)
                else:
                    cnf_obj = PBEnc.equals(lits, weights, rhs, top_id=top_id)

            if cnf_obj:
                output_clauses.extend(cnf_obj.clauses)
                curr_max = top_id
                for c in cnf_obj.clauses:
                    for l in c:
                        curr_max = max(curr_max, abs(l))
                top_id = curr_max
        
        return output_clauses


class MiniZincCoreBackend(IRBackend):
    @property
    def name(self) -> str:
        return "mzn_core"

    def allowed_kinds(self) -> Dict[str, Any]:
        return {
            "exactly_one": {"parameters": {"vars": "List[str]"}},
            "alldifferent_onehot": {"parameters": {"groups": "List[List[str]]"}},
            "linear_leq": {"parameters": {"terms": "List[Dict]", "rhs": "int"}},
            "linear_eq": {"parameters": {"terms": "List[Dict]", "rhs": "int"}}
        }

    def validate_constraint(self, constraint: ModelingConstraint, state: AgentState) -> None:
        if constraint.kind not in self.allowed_kinds(): raise ValueError(f"Unknown kind {constraint.kind}")
        p = constraint.parameters
        if constraint.kind == "exactly_one":
            if "vars" not in p or not isinstance(p["vars"], list): raise ValueError("missing 'vars' list")
            for v in p["vars"]:
                if str(v).lstrip('-~') not in state.sat_variables: raise ValueError(f"Var {v} not found")
        elif constraint.kind == "alldifferent_onehot":
            if "groups" not in p or not isinstance(p["groups"], list) or not p["groups"]:
                raise ValueError("requires non-empty 'groups' list")
            # Enforce strict equal length
            first_len = len(p["groups"][0])
            for i, g in enumerate(p["groups"]):
                if not isinstance(g, list) or not g: raise ValueError(f"Group {i} invalid")
                if len(g) != first_len: raise ValueError(f"Group {i} length mismatch")
                for v in g:
                    if str(v).lstrip('-~') not in state.sat_variables: raise ValueError(f"Var {v} not found")
        elif constraint.kind in ["linear_leq", "linear_eq"]:
            if "terms" not in p or "rhs" not in p: raise ValueError("requires 'terms', 'rhs'")
            for t in p["terms"]:
                if str(t["var"]).lstrip('-~') not in state.sat_variables: raise ValueError(f"Var {t['var']} not found")

    def compile_constraints(self, constraints: List[ModelingConstraint], state: AgentState) -> List[List[int]]:
        output_clauses = []
        top_id = state.next_var_id - 1
        
        def get_lit(name):
            name_str = str(name)
            is_neg = name_str.startswith('-') or name_str.startswith('~')
            atom = name_str.lstrip('-~')
            vid = state.sat_variables.get(atom)
            if vid is None: raise ValueError(f"Missing var {atom}")
            return -vid if is_neg else vid

        for constr in constraints:
            if constr.ir_backend != self.name: continue
            k = constr.kind
            p = constr.parameters
            
            cnf_obj = None
            if k == "exactly_one":
                 cnf_obj = CardEnc.equals([get_lit(v) for v in p["vars"]], 1, top_id=top_id)
            elif k == "alldifferent_onehot":
                groups = p["groups"]
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        g1, g2 = groups[i], groups[j]
                        # Validated equal length, can iterate using either len
                        for idx in range(len(g1)):
                            output_clauses.append([-get_lit(g1[idx]), -get_lit(g2[idx])])
            elif k in ["linear_leq", "linear_eq"]:
                lits = [get_lit(t["var"]) for t in p["terms"]]
                weights = [int(t["coef"]) for t in p["terms"]]
                rhs = int(p["rhs"])
                if k == "linear_leq":
                    cnf_obj = PBEnc.leq(lits, weights, rhs, top_id=top_id)
                else:
                    cnf_obj = PBEnc.equals(lits, weights, rhs, top_id=top_id)

            if cnf_obj:
                output_clauses.extend(cnf_obj.clauses)
                curr_max = top_id
                for c in cnf_obj.clauses:
                    for l in c:
                        curr_max = max(curr_max, abs(l))
                top_id = curr_max
        
        return output_clauses


class IRBackendRegistry:
    def __init__(self):
        self._backends = {}
        self.register(CNFBackend())
        self.register(PBBackend())
        self.register(MiniZincCoreBackend())

    def register(self, backend: IRBackend):
        self._backends[backend.name] = backend

    def get(self, name: str) -> IRBackend:
        if name not in self._backends:
            raise ValueError(f"Backend '{name}' not found.")
        return self._backends[name]

    def list_backends(self) -> List[str]:
        return list(self._backends.keys())


# ==============================================================================
# ACTIONS & MANAGER
# ==============================================================================

class ActionType(str, Enum):
    DEFINE_VARIABLES = "DEFINE_VARIABLES"
    ADD_MODEL_CONSTRAINTS = "ADD_MODEL_CONSTRAINTS"
    REMOVE_MODEL_CONSTRAINTS = "REMOVE_MODEL_CONSTRAINTS"
    LIST_IR_SCHEMA = "LIST_IR_SCHEMA"
    SOLVE = "SOLVE"
    TEST_CONSTRAINT = "TEST_CONSTRAINT"
    FUZZ_CONSTRAINTS = "FUZZ_CONSTRAINTS"
    DECODE_SOLUTION = "DECODE_SOLUTION"
    FINISH = "FINISH"
    # Legacy
    ADD_CONSTRAINTS = "ADD_CONSTRAINTS" 
    REFINE_MODEL = "REFINE_MODEL"


class SATManager:
    def __init__(self, registry: Optional[IRBackendRegistry] = None):
        self.solver_name = 'g3'
        self.registry = registry if registry else IRBackendRegistry()

    def _compile(self, state: AgentState):
        """JIT compilation updating state.cnf_clauses and state.next_var_id safely."""
        backend = self.registry.get(state.active_ir_backend)
        valid = []
        for c in state.model_constraints:
            if c.ir_backend != state.active_ir_backend:
                raise ValueError(f"Mixed backend constraint {c.id}")
            backend.validate_constraint(c, state)
            valid.append(c)
        
        state.cnf_clauses = backend.compile_constraints(valid, state)
        
        # Sanity Guard
        for i, c in enumerate(state.cnf_clauses):
            if not c: raise ValueError(f"Empty clause produced at index {i}")
            for l in c:
                if l == 0: raise ValueError(f"Zero literal in clause {i}")

        # safely update next_var_id watermark
        max_used = 0
        if state.sat_variables:
            max_used = max(state.sat_variables.values())
        for c in state.cnf_clauses:
            for l in c:
                max_used = max(max_used, abs(l))
        state.next_var_id = max_used + 1

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
    def define_variables(self, state: AgentState, var_names: List[str]) -> str:
        added = []
        for name in var_names:
            if name not in state.sat_variables:
                state.sat_variables[name] = state.next_var_id
                state.next_var_id += 1
                added.append(name)
        return f"Registered {len(added)} variables: {added}"

    def add_model_constraints(self, state: AgentState, constraints_data: List[Dict[str, Any]]) -> str:
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

                c_id = c_data.get("id", str(uuid.uuid4()))
                c = ModelingConstraint(id=c_id, ir_backend=backend.name, kind=kind, parameters=c_data["parameters"])
                backend.validate_constraint(c, state)
                state.model_constraints.append(c)
                added_count += 1
        except Exception as e:
            return f"Error: {str(e)}"
        return f"Added {added_count} constraints."

    def remove_model_constraints(self, state: AgentState, ids: List[str]) -> str:
        orig = len(state.model_constraints)
        state.model_constraints = [c for c in state.model_constraints if c.id not in ids]
        return f"Removed {orig - len(state.model_constraints)} constraints."

    def get_schema(self, state: AgentState) -> str:
        try:
            return json.dumps(self.registry.get(state.active_ir_backend).allowed_kinds())
        except: return "Error"

    def add_constraints(self, state: AgentState, constraints: List[List[str]]) -> str:
        return "Error: Deprecated"

    def solve(self, state: AgentState) -> str:
        try:
            self._compile(state)
            solver = Solver(name=self.solver_name, bootstrap_with=state.cnf_clauses)
            sat = solver.solve()
            if sat:
                m = solver.get_model()
                solver.delete()
                state.final_status = "SAT"
                return f"SAT. Model with {len(m)} vars."
            else:
                solver.delete()
                state.final_status = "UNSAT"
                return "UNSAT."
        except Exception as e: return f"Error: {e}"

    def test_constraint(self, state: AgentState, assumptions: List[str]) -> str:
        try:
            self._compile(state)
            res = self.solve_cnf_under_assumptions(state.cnf_clauses, assumptions, state)
            if res == "ERROR_VAR": return "Error: Unknown variable"
            return res
        except Exception as e: return f"Error: {e}"

    def refine_model(self, state: AgentState, indices: List[int]) -> str:
        try:
            for idx in sorted(indices, reverse=True):
                if 0 <= idx < len(state.model_constraints): state.model_constraints.pop(idx)
            return f"Removed {len(indices)}."
        except: return "Error"

    def decode_solution(self, state: AgentState) -> str:
        if state.final_status == "UNSAT": return "UNSAT"
        try:
            self._compile(state)
            s = Solver(name=self.solver_name, bootstrap_with=state.cnf_clauses)
            if s.solve():
                m = s.get_model()
                sol = {}
                i2n = {v:k for k,v in state.sat_variables.items()}
                for v in m:
                    if abs(v) in i2n: sol[i2n[abs(v)]] = (v>0)
                state.solution = sol
                s.delete()
                return f"Solution: {json.dumps(sol)}"
            s.delete()
            return "Unexpected UNSAT"
        except Exception as e: return f"Error: {e}"



SYSTEM_PROMPT = """You are an expert SAT Modeler and ReAct Agent.
Your goal is to solve the user's problem by:
1. Decomposing it into boolean variables.
2. Adding constraints using the available schema.
3. Compiling and Solving.
4. If valid, decoding the solution.

You operate in a strict loop:
1. You receive history and context.
2. You output a JSON object with:
   - "thought": string reasoning
   - "action": string enum value
   - "action_input": any or dict

Output strictly JSON.
"""

class ReActAgent:
    def __init__(self, llm_callable, max_steps: int = 20):
        self.llm = llm_callable
        self.max_steps = max_steps
        self.registry = IRBackendRegistry()
        self.sat = SATManager(self.registry)
        self.config = IRConfig.from_env_or_file() 

    def run(self, goal: str) -> AgentState:
        state = AgentState()
        state.active_ir_backend = self.config.backend 
        while not state.finished and state.step_count < self.max_steps:
            state.step_count += 1
            print(f"Step {state.step_count}...", end="\r", flush=True)
            history_prompt = self._construct_prompt(goal, state)
            raw = self.llm(history_prompt)
            try:
                resp = self._parse_llm_output(raw)
            except Exception as e:
                self._record(state, "ParseFail", "ERROR", f"{e} || RAW: {raw}")
                continue
            
            obs = self._execute(state, resp["action"], resp["action_input"])
            self._record(state, resp.get("thought",""), f"{resp['action'].value}", obs)
            if resp["action"] == ActionType.FINISH: state.finished = True
        return state

    def _execute(self, state: AgentState, action: ActionType, arg: Any) -> str:
        try:
            if action == ActionType.DEFINE_VARIABLES: return self.sat.define_variables(state, arg)
            elif action == ActionType.ADD_MODEL_CONSTRAINTS: return self.sat.add_model_constraints(state, arg)
            elif action == ActionType.REMOVE_MODEL_CONSTRAINTS: return self.sat.remove_model_constraints(state, arg)
            elif action == ActionType.LIST_IR_SCHEMA: return self.sat.get_schema(state)
            elif action == ActionType.SOLVE: return self.sat.solve(state)
            elif action == ActionType.TEST_CONSTRAINT: return self.sat.test_constraint(state, arg)
            elif action == ActionType.FUZZ_CONSTRAINTS: 
                if not isinstance(arg, dict): return "Error: Input must be dict"
                return self.sat.fuzz_constraints(state, arg)
            elif action == ActionType.DECODE_SOLUTION: return self.sat.decode_solution(state)
            elif action == ActionType.FINISH: return "Terminating"
            elif action == ActionType.ADD_CONSTRAINTS: return "Deprecated"
            elif action == ActionType.REFINE_MODEL: return self.sat.refine_model(state, arg)
            else: return f"Unknown Action {action}"
        except Exception as e: return f"Exec Error: {e}"

    def _parse_llm_output(self, raw: str) -> Dict[str, Any]:
        # Strip Markdown Code Blocks
        clean_raw = raw.strip()
        if clean_raw.startswith("```"):
            clean_raw = clean_raw.split("\n", 1)[1] # remove first line
            if clean_raw.endswith("```"):
                clean_raw = clean_raw[:-3].strip()
        
        data = json.loads(clean_raw)
        if "action" not in data: raise ValueError("No action")
        try: data["action"] = ActionType(data["action"])
        except: raise ValueError(f"Invalid ActionType: {data.get('action')}")
        return data

    def _record(self, state, thought, act, obs):
        state.trajectory.append((thought, act, obs))

    def _construct_prompt(self, goal: str, state: AgentState) -> str:
        prompt = f"{SYSTEM_PROMPT}\nGOAL: {goal}\nACTIVE_BACKEND: {state.active_ir_backend}\n"
        prompt += f"AVAILABLE: {self.registry.list_backends()}\n"
        prompt += f"SCHEMA: {self.sat.get_schema(state)}\n"
        prompt += f"Use FUZZ_CONSTRAINTS to validate IDs.\n"
        
        prompt += "VARIABLES:\n"
        vars_list = list(state.sat_variables.keys())
        # Truncate if too many?
        if len(vars_list) > 100:
             prompt += f"{vars_list[:100]}... (+{len(vars_list)-100} more)\n"
        else:
             prompt += f"{vars_list}\n"

        prompt += "CONSTRAINTS:\n"
        for c in state.model_constraints:
            prompt += f"- {c.id} | {c.kind} | {json.dumps(c.parameters)}\n"
        prompt += "\nHISTORY:\n"
        for i, (t, a, o) in enumerate(state.trajectory):
            prompt += f"{i}. T:{t} A:{a} O:{o}\n"
        return prompt

if __name__ == "__main__":
    def mock_llm(p):
        if "DEFINE_VARIABLES" not in p: return json.dumps({"thought": "init", "action": "DEFINE_VARIABLES", "action_input": ["A", "B"]})
        if "ADD_MODEL_CONSTRAINTS" not in p: return json.dumps({"thought": "add", "action": "ADD_MODEL_CONSTRAINTS", "action_input": [{"kind": "at_most_k", "parameters": {"vars": ["A", "B"], "k": 1}}]})
        # Fuzzing requires ID which we don't have easily in this dumb mock loop without parsing.
        # Just solve and finish.
        if "SOLVE" not in p: return json.dumps({"thought": "solve", "action": "SOLVE", "action_input": {}})
        return json.dumps({"thought": "done", "action": "FINISH", "action_input": {}})
    
    agent = ReActAgent(mock_llm)
    print(agent.run("test"))
