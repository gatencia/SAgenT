from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class TraceEvent(BaseModel):
    """
    A single step in the encoding process.
    """
    kind: str  # e.g., "IR_NODE", "CNF_EMIT"
    payload: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class EncodingTrace(BaseModel):
    """
    Complete trace of how an encoding was built.
    """
    trace_version: str = "1.0"
    entry_id: Optional[str] = None
    events: List[TraceEvent] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
