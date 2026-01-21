from typing import List, Dict, Any
from engine.backends.base import IRBackend
from engine.state import AgentState, ModelingConstraint

class CNFBackend(IRBackend):
    @property
    def name(self) -> str:
        return "cnf"

    def get_prompt_doc(self) -> str:
        return """
### BACKEND: CNF (Raw SAT)
You are using the Raw CNF backend.
This backend acts as a direct pass-through to the SAT solver.
- **Capabilities**: Only supports raw `clause` injection (Disjunction of literals).
- **Mapping**: No high-level abstractions. You must bit-blast everything yourself.
- **Use Case**: Only use this if you want manual control over every clause.
"""

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
