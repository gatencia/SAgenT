import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

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

class AgentPhase(str, Enum):
    PLANNING = "PLANNING"
    IMPLEMENTATION = "IMPLEMENTATION"
    DEBUGGING = "DEBUGGING"

@dataclass
class AgentState:
    trajectory: List[Tuple[str, str, str]] = field(default_factory=list)
    step_count: int = 0
    finished: bool = False
    final_status: Optional[str] = None
    
    # SAT/CSP State
    sat_variables: Dict[str, int] = field(default_factory=dict)
    next_var_id: int = 1
    model_constraints: List[ModelingConstraint] = field(default_factory=list)
    minizinc_code: List[str] = field(default_factory=list) # Raw high-level code
    model_file_path: Optional[str] = None # Path to the active file being edited
    
    # Active Backend
    active_ir_backend: str = "pb"
    
    # Planning
    plan: Optional[Dict[str, Any]] = None # {observations, variables, constraints, strategy, verification}
    current_phase: AgentPhase = AgentPhase.PLANNING # Strict Phase Control

    # Execution
    cnf_clauses: List[List[int]] = field(default_factory=list)
    solution: Optional[Dict[str, bool]] = None
    validator_results: List[Dict[str, Any]] = field(default_factory=list)
    
    fuzz_log: List[Dict[str, Any]] = field(default_factory=list)
    compile_report: Optional[Dict[str, Any]] = None

    def serialize(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
