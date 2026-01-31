import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from engine.vars import VarManager
from engine.compilation.artifact import CompilationArtifact
from engine.solution.types import SatResult, DomainSolution

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
    var_manager: VarManager = field(default_factory=VarManager)
    compilation_artifact: Optional[CompilationArtifact] = None
    
    cnf_clauses: List[List[int]] = field(default_factory=list)
    solution: Optional[Dict[str, bool]] = None
    
    # Rich Decode
    sat_result: Optional[SatResult] = None
    domain_solution: Optional[DomainSolution] = None # Decoded objects (paths, grid, etc)

    validator_results: List[Dict[str, Any]] = field(default_factory=list)
    
    fuzz_log: List[Dict[str, Any]] = field(default_factory=list)
    compile_report: Optional[Dict[str, Any]] = None
    
    # Telemetry & Research Logs
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "syntax_errors": 0,
        "schema_violations": 0,
        "rejections": 0,
        "iterations_to_valid": 0,
        "fuzz_failures": 0,
        "unsat_cores_trigger": 0,
        "first_shot_success": False
    })
    run_logs: List[Dict[str, Any]] = field(default_factory=list) # Timeline of events

    def serialize(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
