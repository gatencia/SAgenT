from pydantic import BaseModel
from typing import List, Optional
from Denabase.core.types import CNFMetadata, CNFStats, VerificationRecord

class DBEntry(BaseModel):
    """A record in the Denabase."""
    id: str
    metadata: CNFMetadata
    stats: CNFStats
    embedding: List[float]
    verification: Optional[VerificationRecord] = None
    cnf_path: str
