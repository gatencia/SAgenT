"""
Core module for Denabase.
Provides error handling, logging, types, and serialization.
"""
from Denabase.Denabase.core.errors import (
    DenabaseError, ValidationError, StorageError, CNFError,
    IRCompileError, VerificationError, EmbeddingError, IndexError, SelectionError
)
from Denabase.Denabase.core.logging import get_logger
from Denabase.Denabase.core.types import (
    CnfVar, Lit, Clause, CNF, validate_cnf, sha256_bytes, utc_now_iso
)
from Denabase.Denabase.core.serialization import (
    to_json, from_json, atomic_write_text, atomic_write_bytes, safe_mkdir, read_text, read_bytes
)

__all__ = [
    "DenabaseError", "ValidationError", "StorageError", "CNFError",
    "IRCompileError", "VerificationError", "EmbeddingError", "IndexError", "SelectionError",
    "get_logger",
    "CnfVar", "Lit", "Clause", "CNF", "validate_cnf", "sha256_bytes", "utc_now_iso",
    "to_json", "from_json", "atomic_write_text", "atomic_write_bytes", "safe_mkdir", "read_text", "read_bytes"
]
