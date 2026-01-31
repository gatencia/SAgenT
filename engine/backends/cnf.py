from typing import List, Dict, Any
from engine.backends.base import IRBackend
from engine.state import AgentState, ModelingConstraint
from engine.compilation.artifact import CompilationArtifact

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

    def compile(self, state: AgentState) -> CompilationArtifact:
        """
        Compiles constraints into CompilationArtifact using VarManager.
        """
        output_clauses = []
        constraint_to_clause_ids = {}
        
        backend_stats = {"constraints_processed": 0}
        
        for constr in state.model_constraints:
            current_clauses = []
            if constr.ir_backend == self.name and constr.kind == "clause":
                clause_ints = []
                for lit in constr.parameters["literals"]:
                    str_lit = str(lit)
                    is_neg = str_lit.startswith('-') or str_lit.startswith('~') or str_lit.startswith('!')
                    atom = str_lit.lstrip('-~!')
                    
                    # Resolve via VarManager
                    try:
                        vid = state.var_manager.declare(atom)
                    except Exception:
                        if atom in state.sat_variables:
                            vid = state.sat_variables[atom]
                        else:
                            vid = state.var_manager.declare(atom)
                            
                    clause_ints.append(-vid if is_neg else vid)
                    
                if clause_ints:
                    current_clauses.append(clause_ints)
                    
            start_idx = len(output_clauses)
            output_clauses.extend(current_clauses)
            end_idx = len(output_clauses)
            
            constraint_to_clause_ids[constr.id] = list(range(start_idx, end_idx))
            backend_stats["constraints_processed"] += 1

        return CompilationArtifact(
            backend_name=self.name,
            encoding_config={},
            clauses=output_clauses,
            var_map=state.var_manager.get_var_map(),
            id_to_name=state.var_manager.get_id_to_name(),
            constraint_ids=[c.id for c in state.model_constraints],
            constraint_to_clause_ids=constraint_to_clause_ids,
            aux_vars=set(), # CNF backend doesn't typically allocate aux vars implicitly
            stats=backend_stats
        )

    def compile_constraints(self, constraints: List[ModelingConstraint], state: AgentState) -> List[List[int]]:
        output_clauses = []
        for constr in constraints:
            if constr.ir_backend == self.name and constr.kind == "clause":
                clause_ints = []
                for lit in constr.parameters["literals"]:
                    is_neg = str(lit).startswith('-') or str(lit).startswith('~') or str(lit).startswith('!')
                    atom = str(lit).lstrip('-~!')
                    if atom in state.sat_variables:
                        var_id = state.sat_variables[atom]
                        clause_ints.append(-var_id if is_neg else var_id)
                if clause_ints:
                    output_clauses.append(clause_ints)
        return output_clauses

    def generate_code(self, state: AgentState) -> str:
        """Return a verbose summary of the CNF clauses."""
        lines = []
        lines.append("% CNF Formulation (Auto-Generated)")
        lines.append(f"% Variables: {len(state.sat_variables)}")
        lines.append("")
        
        # We try to use high-level constraints if they exist (source of truth)
        # Even for CNF backend, the user ADDS constraints via 'clause' kind.
        for c in state.model_constraints:
            if c.kind == "clause":
                lits = c.parameters.get("literals", [])
                lines.append(f"clause({', '.join(str(l) for l in lits)});")
            else:
                lines.append(f"% Unknown/Unsupported kind for CNF: {c.kind}")
        
        return "\n".join(lines)
