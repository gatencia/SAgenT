from typing import List, Dict, Any
from engine.backends.base import IRBackend
from engine.state import AgentState, ModelingConstraint
from engine.connectivity.registry import ConnectivityRegistry

# PySAT imports
try:
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

    def generate_code(self, state: AgentState) -> str:
        """Return a representation of the Pseudo-Boolean state."""
        return f"Pseudo-Boolean Backend Summary:\nActive Constraints: {len(state.model_constraints)}"
