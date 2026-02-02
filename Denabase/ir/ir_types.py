import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator

# --- VarRef ---

class VarRef(BaseModel):
    """Reference to a named variable with strict validation."""
    name: str

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("Variable name cannot be empty")
        if len(v) > 64:
            raise ValueError("Variable name too long (max 64)")
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Variable name must be alphanumeric/underscore")
        if " " in v:
            raise ValueError("Variable name cannot contain spaces")
        return v

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, VarRef):
            return False
        return self.name == other.name

# --- BoolExpr ---

class BoolExprBase(BaseModel):
    pass

class Lit(BoolExprBase):
    kind: Literal["lit"] = "lit"
    var: VarRef
    neg: bool = False

class Not(BoolExprBase):
    kind: Literal["not"] = "not"
    term: "BoolExpr"

class And(BoolExprBase):
    kind: Literal["and"] = "and"
    terms: List["BoolExpr"]

    @field_validator('terms')
    @classmethod
    def validate_len(cls, v: List[Any]) -> List[Any]:
        if len(v) < 2:
            raise ValueError("And requires at least 2 terms")
        return v

class Or(BoolExprBase):
    kind: Literal["or"] = "or"
    terms: List["BoolExpr"]

    @field_validator('terms')
    @classmethod
    def validate_len(cls, v: List[Any]) -> List[Any]:
        if len(v) < 2:
            raise ValueError("Or requires at least 2 terms")
        return v

class Imp(BoolExprBase):
    kind: Literal["imp"] = "imp"
    a: "BoolExpr"
    b: "BoolExpr"

class Iff(BoolExprBase):
    kind: Literal["iff"] = "iff"
    a: "BoolExpr"
    b: "BoolExpr"

class Xor(BoolExprBase):
    kind: Literal["xor"] = "xor"
    a: "BoolExpr"
    b: "BoolExpr"

BoolExpr = Annotated[
    Union[Lit, Not, And, Or, Imp, Iff, Xor],
    Field(discriminator="kind")
]

# Required for recursive models in Pydantic v2
Not.model_rebuild()
And.model_rebuild()
Or.model_rebuild()
Imp.model_rebuild()
Iff.model_rebuild()
Xor.model_rebuild()

# --- Cardinality ---

class CardinalityBase(BaseModel):
    k: int
    vars: List[VarRef]

    @field_validator('vars')
    @classmethod
    def validate_unique(cls, v: List[VarRef]) -> List[VarRef]:
        names = [vr.name for vr in v]
        if len(names) != len(set(names)):
            raise ValueError("Variable names must be unique in cardinality constraint")
        return v

    @model_validator(mode='after')
    def validate_k_range(self) -> 'CardinalityBase':
        if not (0 <= self.k <= len(self.vars)):
            raise ValueError(f"k must be in range [0, {len(self.vars)}]")
        return self

class AtLeast(CardinalityBase):
    kind: Literal["at_least"] = "at_least"

class AtMost(CardinalityBase):
    kind: Literal["at_most"] = "at_most"

class Exactly(CardinalityBase):
    kind: Literal["exactly"] = "exactly"

Cardinality = Annotated[
    Union[AtLeast, AtMost, Exactly],
    Field(discriminator="kind")
]

class FixedCNF(BaseModel):
    """
    Represents a fixed CNF fragment (e.g. from a learned gadget fallback).
    """
    kind: Literal["fixed_cnf"] = "fixed_cnf"
    clauses: List[List[int]] # Raw DIMACS-style clauses (internal vars 1..N)
    vars: List[VarRef] # Positional ports mapping to internal vars 1..len(vars)

IR = RootModel[Union[BoolExpr, Cardinality, FixedCNF, List[Union[BoolExpr, Cardinality, FixedCNF]]]]
