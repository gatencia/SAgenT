import hashlib
from typing import List, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from pysat.formula import CNF
from Denabase.Denabase.core.types import sha256_bytes

# Type alias for backward compatibility with skeleton code
CNFEncoding = List[List[int]]

def canonicalize_clause(clause: List[int]) -> Tuple[int, ...]:
    """Sorts literals in a clause by absolute value, then sign."""
    return tuple(sorted(clause, key=lambda x: (abs(x), x)))

def canonicalize_cnf(clauses: List[List[int]]) -> List[Tuple[int, ...]]:
    """Sorts clauses lexicographically after canonicalizing each clause."""
    canonical_clauses = [canonicalize_clause(c) for c in clauses]
    return sorted(canonical_clauses)

class CnfDocument(BaseModel):
    """Production-grade CNF document model with validation."""
    num_vars: int = Field(ge=0)
    clauses: List[List[int]]

    @field_validator('clauses')
    @classmethod
    def validate_clauses(cls, v: List[List[int]], info) -> List[List[int]]:
        num_vars = info.data.get('num_vars')
        for i, clause in enumerate(v):
            if not clause:
                raise ValueError(f"Clause {i} is empty")
            for lit in clause:
                if lit == 0:
                    raise ValueError(f"Literal 0 is invalid in clause {i}")
                if num_vars is not None and abs(lit) > num_vars:
                    raise ValueError(f"Literal {lit} exceeds num_vars {num_vars} in clause {i}")
        return v

    def to_pysat(self) -> CNF:
        """Converts to a PySAT CNF formula."""
        formula = CNF()
        formula.nv = self.num_vars
        formula.clauses = [list(c) for c in self.clauses]
        return formula

    @classmethod
    def from_pysat(cls, formula: CNF) -> 'CnfDocument':
        """Creates a CnfDocument from a PySAT CNF formula."""
        return cls(num_vars=formula.nv, clauses=[list(c) for c in formula.clauses])

    def get_canonical_clauses(self) -> List[Tuple[int, ...]]:
        """Returns the canonicalized clauses."""
        return canonicalize_cnf(self.clauses)

    def content_hash(self) -> str:
        """Computes a stable SHA256 hash of the canonicalized CNF content."""
        canonical = self.get_canonical_clauses()
        # Stable string representation for hashing
        content = f"p cnf {self.num_vars} {len(canonical)}\n"
        content += "\n".join(" ".join(map(str, c)) + " 0" for c in canonical)
        return sha256_bytes(content.encode('utf-8'))
