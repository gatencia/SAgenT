import abc
from typing import List, Dict, Any
from engine.state import AgentState, ModelingConstraint

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

    @abc.abstractmethod
    def generate_code(self, state: AgentState) -> str:
        """Return the actual code/representation that would be sent to the solver."""
        pass

    def get_prompt_doc(self) -> str:
        """Return specific instructions/hints for the LLM on how to use this backend."""
        return f"Backend {self.name}: No specific instructions provided."
