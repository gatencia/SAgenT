import abc
from typing import List, Dict, Any
from engine.state import AgentState, ModelingConstraint
from engine.compilation.artifact import CompilationArtifact

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

    def compile_constraints_with_metadata(self, constraints: List[ModelingConstraint], state: AgentState) -> tuple[List[List[int]], Dict[str, List[List[int]]]]:
        """
        Compiles constraints and returns (all_clauses, metadata_map).
        metadata_map: {constraint_id: list_of_clauses_for_this_constraint}
        Default implementation just loops and calls compile_constraints (inefficient but safe for non-aux backends).
        """
        all_clauses = []
        metadata = {}
        for c in constraints:
            c_clauses = self.compile_constraints([c], state)
            all_clauses.extend(c_clauses)
            metadata[c.id] = c_clauses
        return all_clauses, metadata

    @abc.abstractmethod
    def compile(self, state: AgentState) -> CompilationArtifact:
        """
        Full compilation pass producing a provenance-rich artifact.
        Must use state.var_manager for all allocation.
        """
        pass

    @abc.abstractmethod
    def allowed_kinds(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def generate_code(self, state: AgentState) -> str:
        """Return the actual code/representation that would be sent to the solver."""
        pass

    def get_prompt_doc(self) -> str:
        """Return specific instructions/hints for the LLM on how to use this backend."""
        return f"Backend {self.name}: No specific instructions provided."
