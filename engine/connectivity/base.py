import abc
from typing import Dict, Any, List
from engine.state import AgentState, ModelingConstraint

class ConnectivityEncoder(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: pass

    @abc.abstractmethod
    def required_kinds(self) -> Dict[str, Any]: pass

    @abc.abstractmethod
    def validate(self, constraint: ModelingConstraint, state: AgentState) -> None: pass

    @abc.abstractmethod
    def compile(self, constraint: ModelingConstraint, state: AgentState) -> List[List[int]]: pass
