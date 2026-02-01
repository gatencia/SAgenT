from typing import List, Dict, Union, Any, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from Denabase.ir.ir_types import BoolExpr, Cardinality, VarRef, Lit, And, Or, Not, Exactly, AtMost, Iff
from Denabase.core.errors import ValidationError

class GadgetUnitTest(BaseModel):
    """A single unit test case for a gadget."""
    params: Dict[str, Any]
    expected_sat: bool

class GadgetSpec(ABC):
    """Abstract base class for reusable constraint gadgets."""
    
    name: str
    family: str
    version: str = "1.0"
    description: str
    params_schema: Dict[str, Any] = Field(default_factory=dict)
    unit_tests: List[GadgetUnitTest] = Field(default_factory=list)

    def validate_params(self, params: Dict[str, Any]):
        """Simple validation against expected keys/types."""
        # For now, just check required keys. Full JSON schema validation could be added later.
        required = self.params_schema.get("required", [])
        for k in required:
            if k not in params:
                raise ValidationError(f"Gadget {self.name} missing required param: {k}")

    @abstractmethod
    def build_ir(self, params: Dict[str, Any]) -> Union[BoolExpr, Cardinality, List[Union[BoolExpr, Cardinality]]]:
        """Builds the IR for the gadget given the parameters."""
        pass

# --- Built-in Gadgets ---

class ExactlyOneGadget(GadgetSpec):
    name = "exactly_one"
    family = "cardinality"
    description = "Enforces exactly one of the given variables is true."
    params_schema = {
        "type": "object",
        "properties": {
            "vars": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["vars"]
    }
    unit_tests = [
        GadgetUnitTest(params={"vars": ["a", "b"]}, expected_sat=True),
        # Testing specific models requires more than just expected_sat (needs decoder), 
        # but for simple self-test we rely on verifier to check ANY SAT.
        # Ideally, unit test should say "params X should be SAT/UNSAT". 
        # For ExactlyOne, it is satisfiable if len > 0.
        GadgetUnitTest(params={"vars": ["x"]}, expected_sat=True)
    ]

    def build_ir(self, params: Dict[str, Any]) -> Cardinality:
        self.validate_params(params)
        varnames = params["vars"]
        refs = [VarRef(name=v) for v in varnames]
        return Exactly(k=1, vars=refs)

class AtMostOneGadget(GadgetSpec):
    name = "at_most_one"
    family = "cardinality"
    description = "Enforces at most one of the given variables is true."
    params_schema = {
        "type": "object",
        "properties": {
            "vars": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["vars"]
    }
    unit_tests = [
        GadgetUnitTest(params={"vars": ["a", "b", "c"]}, expected_sat=True)
    ]

    def build_ir(self, params: Dict[str, Any]) -> Cardinality:
        self.validate_params(params)
        varnames = params["vars"]
        refs = [VarRef(name=v) for v in varnames]
        return AtMost(k=1, vars=refs)

class KColoringGadget(GadgetSpec):
    name = "k_coloring"
    family = "graph"
    description = "K-Coloring constraint for a graph."
    params_schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "k": {"type": "integer"},
            "edges": {"type": "array", "items": {"type": "array"}}
        },
        "required": ["n", "k", "edges"]
    }
    # Triangle K=2 -> UNSAT
    # Triangle K=3 -> SAT
    unit_tests = [
        GadgetUnitTest(
            params={"n": 3, "k": 2, "edges": [[0, 1], [1, 2], [2, 0]]}, 
            expected_sat=False
        ),
        GadgetUnitTest(
            params={"n": 3, "k": 3, "edges": [[0, 1], [1, 2], [2, 0]]}, 
            expected_sat=True
        )
    ]

    def build_ir(self, params: Dict[str, Any]) -> List[Union[BoolExpr, Cardinality]]:
        self.validate_params(params)
        n = params["n"]
        k = params["k"]
        edges = params["edges"]
        
        constraints = []
        
        # 1. Each node must have exactly one color
        for i in range(n):
            node_vars = [VarRef(name=f"n{i}_c{c}") for c in range(k)]
            constraints.append(Exactly(k=1, vars=node_vars))
            
        # 2. Adjacent nodes cannot have same color
        for u, v in edges:
            for c in range(k):
                # NOT (u_c AND v_c) -> Or(Not(u_c), Not(v_c))
                # Or just Not(And(...))
                lu = VarRef(name=f"n{u}_c{c}")
                lv = VarRef(name=f"n{v}_c{c}")
                
                # conflict: not (u has c AND v has c)
                constraints.append(
                    Not(term=And(terms=[Lit(var=lu), Lit(var=lv)]))
                )
                
        return constraints
