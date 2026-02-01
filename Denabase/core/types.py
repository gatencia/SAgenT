import hashlib
from datetime import datetime, timezone
from typing import Annotated, List, Dict, Any, Union
from pydantic import BaseModel, Field, RootModel, field_validator, AfterValidator
from Denabase.core.errors import ValidationError

# Scalar types and constraints
def check_nonzero(v: int) -> int:
    if v == 0:
        raise ValueError("Literal cannot be zero")
    return v

CnfVar = Annotated[int, Field(gt=0)]
Lit = Annotated[int, AfterValidator(check_nonzero)]

# Structural types for CNF
class Clause(RootModel):
    """A non-empty list of literals."""
    root: List[Lit]

    @field_validator('root')
    @classmethod
    def check_not_empty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("Clause must not be empty")
        return v

class CNF(RootModel):
    """A list of clauses."""
    root: List[Clause]

# Database and Metadata types
class CNFMetadata(BaseModel):
    """Metadata for a CNF problem instance."""
    family: str
    problem_id: str
    tags: List[str] = Field(default_factory=list)
    custom: Dict[str, Any] = Field(default_factory=dict)

class CNFStats(BaseModel):
    """Basic statistics for a CNF file."""
    n_vars: int
    n_clauses: int
    clause_len: Dict[str, Union[int, float]]  # min, mean, max
    polarity_ratio: float
    var_occurrence: Dict[str, Union[int, float]] # min, mean, max, gini
    clause_size_histogram: Dict[Union[int, str], int] # keys 1..10 and "overflow"

class VerificationRecord(BaseModel):
    """Record of a verification run."""
    timestamp: str
    passed: bool
    method: str
    checksum: str

# Utilities
def validate_cnf(cnf: List[List[int]]) -> None:
    """
    Validates a CNF structure.
    Raises ValidationError if the structure is invalid.
    """
    try:
        # Wrap in Pydantic models for validation
        CNF.model_validate(cnf)
    except Exception as e:
        raise ValidationError(f"Invalid CNF structure: {e}")

def sha256_bytes(data: bytes) -> str:
    """Returns the SHA256 hash of bytes as a hex string."""
    return hashlib.sha256(data).hexdigest()

def utc_now_iso() -> str:
    """Returns the current UTC time in ISO8601 format with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
