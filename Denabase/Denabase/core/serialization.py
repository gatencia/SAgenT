import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

def to_json(obj: Any) -> str:
    """Serializes an object to a stable JSON string (sorted keys, no NaN)."""
    return json.dumps(obj, sort_keys=True, allow_nan=False, indent=2)

def from_json(s: str) -> Any:
    """Deserializes an object from a JSON string."""
    return json.loads(s)

def atomic_write_text(path: Path, text: str) -> None:
    """Writes text to a file atomically using a temporary file."""
    path = Path(path)
    safe_mkdir(path.parent)
    
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.tmp")
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Writes bytes to a file atomically using a temporary file."""
    path = Path(path)
    safe_mkdir(path.parent)
    
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.tmp")
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def safe_mkdir(path: Path) -> None:
    """Ensures a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def read_text(path: Path) -> str:
    """Reads text from a file with explicit UTF-8 handling."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_bytes(path: Path) -> bytes:
    """Reads bytes from a file."""
    with open(path, "rb") as f:
        return f.read()
