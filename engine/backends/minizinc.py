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
