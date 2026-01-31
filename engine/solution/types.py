from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

class SatStatus(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"

@dataclass
class SatResult:
    """
    Raw result from the SAT solver backend.
    """
    status: SatStatus
    # Raw boolean model: var_literal -> bool (derived from var_map)
    # Ideally should map str -> bool for cleaner API?
    # Keeping it simple: var_name -> bool
    model: Optional[Dict[str, bool]] = None
    
    # Conflict/Core info for UNSAT
    unsat_core: List[str] = field(default_factory=list) # Group IDs
    conflict_info: Dict[str, Any] = field(default_factory=dict) # Metadata
    
    # Perf
    time_taken: float = 0.0

@dataclass
class DomainSolution:
    """
    High-level, domain-specific interpretation of the SAT result.
    """
    is_valid: bool
    
    # Generic map of active variables
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # MRPP Specifics (Optional, should ideally be generic or subclassed)
    # But for this task, we put it here or in a 'details' dict
    paths: Optional[Dict[str, List[any]]] = None
    grid: Optional[Any] = None
    
    raw_result: Optional[SatResult] = None
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "variables": self.variables,
            "paths": self.paths,
            "grid": self.grid
        }
