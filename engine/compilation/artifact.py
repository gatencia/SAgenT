from dataclasses import dataclass, field
from typing import List, Dict, Any, Set

@dataclass
class CompilationArtifact:
    """
    Represents the result of a single compilation pass of the model.
    This artifact contains the full CNF encoding and metadata for debugging.
    """
    backend_name: str
    encoding_config: Dict[str, Any]
    
    # The primary SAT payload
    clauses: List[List[int]]
    
    # Metadata for provenance and decoding
    var_map: Dict[str, int]               # name -> sat_id
    id_to_name: Dict[int, str]            # sat_id -> name
    
    # Grouping info
    constraint_ids: List[str]             # List of high-level constraint IDs involved
    constraint_to_clause_ids: Dict[str, List[int]]  # constraint_id -> list of indices in `clauses`
    
    # Auxiliary variables allocated during compilation
    aux_vars: Set[int]
    
    # Stats
    stats: Dict[str, Any] = field(default_factory=dict)
