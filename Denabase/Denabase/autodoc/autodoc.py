from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from Denabase.Denabase.ir.ir_types import Exactly, AtMost, AtLeast, BoolExpr, Lit, Not, And, Or, Imp, Iff, Xor

class GadgetDoc(BaseModel):
    category: str
    description: str
    params: Dict[str, Any] = {}

class AutoDoc:
    """
    Deterministically summarizes a gadget based on its IR structure.
    """
    
    def summarize(self, ir_obj: Any) -> GadgetDoc:
        if isinstance(ir_obj, list):
            # Compound
            return GadgetDoc(
                category="Compound",
                description=f"A collection of {len(ir_obj)} constraints.",
                params={"count": len(ir_obj)}
            )
            
        # Cardinality
        if isinstance(ir_obj, Exactly):
            n = len(ir_obj.vars)
            k = ir_obj.k
            if k == 1:
                return GadgetDoc(
                    category="Cardinality/ExactlyOne",
                    description=f"Constraints exactly one variable to be true out of {n}.",
                    params={"k": k, "n": n}
                )
            else:
                return GadgetDoc(
                    category="Cardinality/ExactlyK",
                    description=f"Constraints exactly {k} variables to be true out of {n}.",
                    params={"k": k, "n": n}
                )
        
        elif isinstance(ir_obj, AtMost):
            n = len(ir_obj.vars)
            k = ir_obj.k
            if k == 1:
                return GadgetDoc(
                    category="Cardinality/AtMostOne",
                    description=f"Constraints at most one variable to be true out of {n}.",
                    params={"k": k, "n": n}
                )
            else:
                return GadgetDoc(
                    category="Cardinality/AtMostK",
                    description=f"Constraints at most {k} variables to be true out of {n}.",
                    params={"k": k, "n": n}
                )
                
        elif isinstance(ir_obj, AtLeast):
            n = len(ir_obj.vars)
            k = ir_obj.k
            if k == 1:
                return GadgetDoc(
                    category="Cardinality/AtLeastOne",
                    description=f"Constraints at least one variable to be true out of {n}.",
                    params={"k": k, "n": n}
                )
            else:
                return GadgetDoc(
                    category="Cardinality/AtLeastK",
                    description=f"Constraints at least {k} variables to be true out of {n}.",
                    params={"k": k, "n": n}
                )

        # Boolean Logic
        kind_map = {
            "and": "Conjunction (AND)",
            "or": "Disjunction (OR)", 
            "not": "Negation (NOT)",
            "imp": "Implication",
            "iff": "Equivalence (IFF)",
            "xor": "Exclusive OR (XOR)",
            "lit": "Literal"
        }
        
        # Pydantic models might need .kind access if strictly typed, 
        # but hasattr check is safer if type is Union.
        # ir_types uses model.kind
        if hasattr(ir_obj, "kind"):
            k = ir_obj.kind
            if k in kind_map:
                return GadgetDoc(
                    category=f"Boolean/{kind_map[k]}",
                    description=f"A boolean {kind_map[k]} expression.",
                    params={}
                )
                
        # Fallback
        return GadgetDoc(
            category="Unknown",
            description="Unknown gadget structure.",
            params={}
        )
