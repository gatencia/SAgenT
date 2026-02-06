import json
from pathlib import Path
from typing import Union, Dict, Any
from Denabase.Denabase.trace.trace_types import EncodingTrace

def save_trace_json(trace: EncodingTrace, path: Union[str, Path]) -> None:
    """Saves a trace to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(trace.model_dump_json(indent=2))

def load_trace_json(path: Union[str, Path]) -> EncodingTrace:
    """Loads a trace from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return EncodingTrace(**data)
