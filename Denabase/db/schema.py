from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from Denabase.core.types import CNFStats

class VerificationRecord(BaseModel):
    """Record of verification details."""
    timestamp: str
    config: Dict[str, Any]
    passed: bool
    checks_run: List[str]
    failures: List[str]
    stats: Dict[str, Any] = Field(default_factory=dict)
    unsat_core_size: Optional[int] = None


class DBMeta(BaseModel):
    """Metadata for the entire Denabase instance."""
    version: str = "0.2.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    counts: Dict[str, int] = Field(default_factory=dict) # e.g. {"entries": 0}
    feature_schema_version: str = "1.0"
    embed_dim: int = 64
    nl_embed_dim: int = 128
    retrieval_alpha_default: float = 0.7
    alpha_model_enabled: bool = True
    alpha_model_version: str = "0.1"
    
    # Dirty flags for deferred rebuilding
    structural_index_dirty: bool = False
    nl_index_dirty: bool = False
    alpha_model_dirty: bool = False

class EntryMeta(BaseModel):
    """Metadata for a specific entry."""
    entry_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    family: str
    problem_id: str
    source: Optional[str] = None
    nl_text: Optional[str] = None
    expected_label: Optional[str] = None # "SAT", "UNSAT"
    split: Optional[str] = None
    dataset_id: Optional[str] = None
    trace_path: Optional[str] = None
    has_trace: bool = False
    tags: List[str] = Field(default_factory=list)
    user_meta: Dict[str, Any] = Field(default_factory=dict)

def canonical_label(lbl: Union[str, bool, int, None]) -> Optional[str]:
    """Normalizes label to SAT or UNSAT."""
    if lbl is None: return None
    if isinstance(lbl, bool):
        return "SAT" if lbl else "UNSAT"
    s = str(lbl).upper().strip()
    if s in ("TRUE", "1", "SAT", "SATISFIABLE"): return "SAT"
    if s in ("FALSE", "0", "UNSAT", "UNSATISFIABLE"): return "UNSAT"
    return None

class EncodingRecipeRecord(BaseModel):
    """Record of the encoding recipe used."""
    cardinality_encoding: str
    aux_policy: str
    var_order: str
    symmetry_breaking: bool
    notes: List[str] = Field(default_factory=list)

class SolverConfigRecord(BaseModel):
    """Record of a solver configuration."""
    solver_name: str
    params: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    
class TelemetryRecord(BaseModel):
    """Telemetry data for a solver run."""
    solver_config: SolverConfigRecord
    duration: float
    outcome: str
    memory_peak: Optional[int] = None

class ProvenanceRecord(BaseModel):
    """Provenance information for an entry."""
    gadgets: List[str] = Field(default_factory=list)
    transforms: List[str] = Field(default_factory=list)
    config_hashes: Dict[str, str] = Field(default_factory=dict)
    
class EntryRecord(BaseModel):
    """Main record for a Denabase entry, pointing to external artifacts."""
    id: str
    meta: EntryMeta
    # Paths relative to db_root
    paths: Dict[str, str] = Field(default_factory=dict)
    # Hashes for integrity
    hashes: Dict[str, str] = Field(default_factory=dict) 
    # Just basic stats in the main record for quick access
    stats_summary: Dict[str, Any] = Field(default_factory=dict)
