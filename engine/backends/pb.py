from typing import List, Dict, Any
from engine.backends.base import IRBackend
from engine.state import AgentState, ModelingConstraint
from engine.connectivity.registry import ConnectivityRegistry
from engine.compilation.artifact import CompilationArtifact

# PySAT imports
try:
    from pysat.formula import CNF
    from pysat.card import CardEnc
    from pysat.pb import PBEnc
except ImportError:
    CardEnc = Any
    PBEnc = Any

class PBBackend(IRBackend):
    def __init__(self):
        self.conn_registry = ConnectivityRegistry()

    @property
    def name(self) -> str:
        return "pb"

    def get_prompt_doc(self) -> str:
        return """
### BACKEND: Pseudo-Boolean (PySAT)
You are using the Pseudo-Boolean backend.
- **Advantages**: Efficient handling of 'at_most_k', 'exactly_one', and 'linear_leq'.
- **Strategy**:
  - Prefer `cardinality` constraints over raw clauses for counting.
  - Usage of `linear_leq` ("sum(terms) <= rhs") is encouraged for resource constraints.
  - Everything compiles to CNF eventually.
"""

    def allowed_kinds(self) -> Dict[str, Any]:
        base = {
            "clause": {"parameters": {"literals": "List[str]"}},
            "implies": {"parameters": {"a": "str", "b": "str"}},
            "at_most_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "at_least_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "exactly_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "exactly_one": {"parameters": {"vars": "List[str]"}},
            "linear_leq": {"parameters": {"terms": "List[Dict{var, coef}]", "rhs": "int"}},
            "linear_eq": {"parameters": {"terms": "List[Dict{var, coef}]", "rhs": "int"}}
        }
        # Inject from Connectivity Registry
        # For now, manually add connected_8 mapped to rank_tree
        rt = self.conn_registry.get_supported_kind("connected_8")
        if rt:
             base.update(rt.required_kinds())
        return base

    def validate_constraint(self, constraint: ModelingConstraint, state: AgentState) -> None:
        kind = constraint.kind
        # Check connectivity
        conn_enc = self.conn_registry.get_supported_kind(kind)
        if conn_enc:
            conn_enc.validate(constraint, state)
            return

        params = constraint.parameters
        if kind == "clause":
            if "literals" not in params or not isinstance(params["literals"], list):
                raise ValueError("clause requires 'literals' list")
            for lit in params["literals"]:
                if str(lit).lstrip('-~') not in state.sat_variables:
                    raise ValueError(f"Var {lit} not found")
        elif kind == "implies":
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

    def compile_constraints_with_metadata(self, constraints: List[ModelingConstraint], state: AgentState) -> tuple[List[List[int]], Dict[str, List[List[int]]]]:
        output_clauses = []
        metadata = {}
        top_id = state.next_var_id - 1
        
        def get_lit(name):
            name_str = str(name)
            is_neg = name_str.startswith('-') or name_str.startswith('~') or name_str.startswith('!')
            atom = name_str.lstrip('-~!')
            vid = state.sat_variables.get(atom)
            if vid is None: raise ValueError(f"Missing var {atom}")
            return -vid if is_neg else vid

        for constr in constraints:
            current_clauses = []
            k = constr.kind
            p = constr.parameters
            
            # Check for direct connection (legacy support)
            if k == "cnf_clauses":
                 # Usually passed as raw list of lists
                 if "clauses" in p:
                     current_clauses.extend(p["clauses"])
            
            elif hasattr(self, "registry") and k in self.registry:
                 # Composite / Higher level abstraction
                 conn_enc = self.registry.get(k)
                 if conn_enc:
                     # Delegate compilation
                     # Note: Composite might need refactor to support metadata
                     # For now, we assume it returns flattened clauses
                     c_cls = conn_enc.compile(constr, state)
                     current_clauses.extend(c_cls)
            
            else:
                cnf_obj = None
                
                if k == "clause":
                    current_clauses.append([get_lit(l) for l in p["literals"]])
                elif k == "implies":
                    current_clauses.append([-get_lit(p["a"]), get_lit(p["b"])])
                elif k == "at_most_k":
                    val_k = int(p["k"])
                    lits = [get_lit(v) for v in p["vars"]]
                    if val_k == 0:
                        current_clauses.extend([[-l] for l in lits])
                    else:
                        cnf_obj = CardEnc.atmost(lits, val_k, top_id=top_id)
                elif k == "at_least_k":
                    cnf_obj = CardEnc.atleast([get_lit(v) for v in p["vars"]], int(p["k"]), top_id=top_id)
            # -- Connectivity Hook --
            conn_enc = self.conn_registry.get_supported_kind(k)
            if conn_enc:
                current_clauses.extend(conn_enc.compile(constr, state))
            
            # -- Standard PB/Card Logic --
            elif k == "clause":
                current_clauses.append([get_lit(l) for l in p["literals"]])
            elif k == "implies":
                current_clauses.append([-get_lit(p["a"]), get_lit(p["b"])])
            elif k == "at_most_k":
                val_k = int(p["k"])
                lits = [get_lit(v) for v in p["vars"]]
                if val_k == 0:
                    current_clauses.extend([[-l] for l in lits])
                else:
                    cnf_obj = CardEnc.atmost(lits, val_k, top_id=top_id)
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
                current_clauses.extend(cnf_obj.clauses)
                curr_max = top_id
                for cl in cnf_obj.clauses:
                    for l in cl:
                        curr_max = max(curr_max, abs(l))
                top_id = curr_max
            
            output_clauses.extend(current_clauses)
            metadata[constr.id] = current_clauses
            
        return output_clauses, metadata

    def compile(self, state: AgentState) -> CompilationArtifact:
        """
        Compiles all constraints into a CompilationArtifact using VarManager.
        """
        output_clauses = []
        metadata = {}
        constraint_to_clause_ids = {}
        aux_vars = set()
        
        # Helper to resolve literals using VarManager
        def get_lit(name):
            name_str = str(name)
            is_neg = name_str.startswith('-') or name_str.startswith('~') or name_str.startswith('!')
            atom = name_str.lstrip('-~!')
            # Try VarManager first
            try:
                vid = state.var_manager.declare(atom) # Declare ensures existence/id
            except Exception:
                # Fallback to sat_variables if var_manager empty (legacy safety)
                if atom in state.sat_variables:
                    vid = state.sat_variables[atom]
                else: 
                     # Only declare if we are sure? 
                     # For now, declare() will create it if missing, which is standard for SAT encoding
                     vid = state.var_manager.declare(atom)
            
            return -vid if is_neg else vid

        # Track start time for stats if desired, skipping for brevity
        
        backend_stats = {"constraints_processed": 0}
        
        for constr in state.model_constraints:
            current_clauses = []
            k = constr.kind
            p = constr.parameters
            
            # Sync top_id from VarManager
            top_id = state.var_manager.max_id
            
            cnf_obj = None
            
            # -- Constraint Logic (mirroring compile_constraints) --
            
            # Trace Logging
            if hasattr(state, "denabase_trace") and state.denabase_trace is not None:
                # Capture sketch step
                try:
                    from Denabase.Denabase.trace import TraceEvent
                    # Normalize Payload
                    # 1. Type
                    t_type = k
                    
                    # 2. k
                    t_k = None
                    if "k" in p: t_k = int(p["k"])
                    elif k == "exactly_one": t_k = 1
                    
                    # 3. Vars & Arity
                    t_vars = []
                    if "vars" in p and isinstance(p["vars"], list):
                        t_vars = [str(v) for v in p["vars"]]
                    elif "literals" in p and isinstance(p["literals"], list):
                        t_vars = [str(l).lstrip("-~!") for l in p["literals"]]
                    elif "terms" in p and isinstance(p["terms"], list):
                        t_vars = [str(t["var"]) for t in p["terms"]]
                    elif "a" in p and "b" in p:
                        t_vars = [str(p["a"]).lstrip("-~!"), str(p["b"]).lstrip("-~!")]
                        
                    t_arity = len(t_vars)
                    
                    # Construct
                    payload = {
                        "type": t_type,
                        "k": t_k,
                        "arity": t_arity,
                        "vars": t_vars,
                        **p
                    }
                    state.denabase_trace.events.append(TraceEvent(kind="IR_NODE", payload=payload))
                except Exception as e: 
                    # print(f"Trace Error: {e}") 
                    pass

            if k == "cnf_clauses":
                 if "clauses" in p: current_clauses.extend(p["clauses"])
            
            elif self.conn_registry.get_supported_kind(k):
                 # Connectivity handling
                 conn_enc = self.conn_registry.get_supported_kind(k)
                 current_clauses.extend(conn_enc.compile(constr, state))
            
            else:
                if k == "clause":
                    current_clauses.append([get_lit(l) for l in p["literals"]])
                elif k == "implies":
                    current_clauses.append([-get_lit(p["a"]), get_lit(p["b"])])
                elif k == "at_most_k":
                    val_k = int(p["k"])
                    lits = [get_lit(v) for v in p["vars"]]
                    if val_k == 0:
                        current_clauses.extend([[-l] for l in lits])
                    else:
                        cnf_obj = CardEnc.atmost(lits, val_k, top_id=top_id)
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
                    current_clauses.extend(cnf_obj.clauses)
                    # Register allocated aux vars
                    # PySAT uses top_id+1 ... cnf_obj.nv
                    new_max = cnf_obj.nv
                    num_new = new_max - top_id
                    if num_new > 0:
                        # Reserve in VarManager to keep in sync
                        # We use prefix="pysat" to indicate source
                        allocated = state.var_manager.reserve_block(num_new, prefix="pysat", namespace="enc")
                        aux_vars.update(allocated)

            # Record provenance
            start_idx = len(output_clauses)
            output_clauses.extend(current_clauses)
            end_idx = len(output_clauses)
            
            # Store indices
            indices = list(range(start_idx, end_idx))
            constraint_to_clause_ids[constr.id] = indices
            backend_stats["constraints_processed"] += 1

        return CompilationArtifact(
            backend_name=self.name,
            encoding_config={},
            clauses=output_clauses,
            var_map=state.var_manager.get_var_map(),
            id_to_name=state.var_manager.get_id_to_name(),
            constraint_ids=[c.id for c in state.model_constraints],
            constraint_to_clause_ids=constraint_to_clause_ids,
            aux_vars=aux_vars,
            stats=backend_stats
        )

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
            
            conn_enc = self.conn_registry.get_supported_kind(k)
            
            if conn_enc:
                 # Delegate compilation
                 output_clauses.extend(conn_enc.compile(constr, state))
                 continue

            if k == "clause":
                output_clauses.append([get_lit(l) for l in p["literals"]])
            elif k == "implies":
                output_clauses.append([-get_lit(p["a"]), get_lit(p["b"])])
            elif k == "at_most_k":
                val_k = int(p["k"])
                lits = [get_lit(v) for v in p["vars"]]
                if val_k == 0:
                     # Edge case: at_most_0 means NONE can be true.
                     # Convert to independent unit clauses [-l] for l in lits
                     output_clauses.extend([[-l] for l in lits])
                     continue
                
                cnf_obj = CardEnc.atmost(lits, val_k, top_id=top_id)
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

    def generate_code(self, state: AgentState) -> str:
        """Return a verbose representation of the Pseudo-Boolean state."""
        lines = []
        lines.append("% Pseudo-Boolean Formulation (Auto-Generated)")
        lines.append(f"% Variables: {len(state.sat_variables)}")
        lines.append("")
        
        for c in state.model_constraints:
            k = c.kind
            p = c.parameters
            
            if k == "clause":
                lines.append(f"clause({p['literals']});")
            elif k == "implies":
                lines.append(f"implies({p['a']} -> {p['b']});")
            elif k in ["at_most_k", "at_least_k", "exactly_k"]:
                lines.append(f"{k}(vars={p['vars']}, k={p['k']});")
            elif k == "exactly_one":
                lines.append(f"exactly_one(vars={p['vars']});")
            elif k in ["linear_leq", "linear_eq"]:
                # Format: 3*x + 2*y <= 5
                terms = []
                for t in p["terms"]:
                    terms.append(f"{t['coef']}*{t['var']}")
                op = "<=" if k == "linear_leq" else "="
                lines.append(f"sum([{', '.join(terms)}]) {op} {p['rhs']};")
            else:
                lines.append(f"% Unknown constraint: {k} {p}")
        
        return "\n".join(lines)
