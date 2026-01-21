import dataclasses
from typing import List, Dict, Any, Optional, Tuple

@dataclasses.dataclass
class ModelingConstraint:
    id: str
    ir_backend: str 
    kind: str       
    parameters: Dict[str, Any]
    category: str = "logic"
    source_text: Optional[str] = None
    compile_report: Optional[Dict[str, Any]] = None

    def serialize(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

@dataclasses.dataclass
class AgentState:
    trajectory: List[Tuple[str, str, str]] = dataclasses.field(default_factory=list)
    sat_variables: Dict[str, int] = dataclasses.field(default_factory=dict)
    next_var_id: int = 1
    model_constraints: List[ModelingConstraint] = dataclasses.field(default_factory=list)
    active_ir_backend: str = "pb"
    cnf_clauses: List[List[int]] = dataclasses.field(default_factory=list)
    step_count: int = 0
    finished: bool = False
    final_status: Optional[str] = None
    solution: Optional[Dict[str, Any]] = None
    fuzz_log: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    validator_results: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    compile_report: Optional[Dict[str, Any]] = None

    def serialize(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
