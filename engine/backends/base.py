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
