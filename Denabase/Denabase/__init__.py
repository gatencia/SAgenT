"""
Denabase: A verified SAT encoding library.
"""
from typing import TYPE_CHECKING

# Lazy import
if TYPE_CHECKING:
    from Denabase.Denabase.db.denabase import DenaBase

def __getattr__(name: str):
    if name == "DenaBase":
        try:
             from Denabase.Denabase.db.denabase import DenaBase
             return DenaBase
        except ImportError as e:
             raise ImportError(f"Failed to import DenaBase. Ensure dependencies (e.g., python-sat) are installed: {e}")
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = ["DenaBase"]
