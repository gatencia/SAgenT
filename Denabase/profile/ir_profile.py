from typing import List, Union
from Denabase.ir.ir_types import (
    BoolExpr, Lit, Not, And, Or, Imp, Iff, Xor,
    Cardinality, AtLeast, AtMost, Exactly
)
from Denabase.profile.profile_types import ConstraintProfile

def compute_ir_profile(obj: Union[BoolExpr, Cardinality, List[Union[BoolExpr, Cardinality]]]) -> ConstraintProfile:
    """Computes a structural profile from IR objects."""
    counts = {
        "and": 0, "or": 0, "not": 0, "imp": 0, "iff": 0, "xor": 0, "lit": 0,
        "at_least": 0, "at_most": 0, "exactly": 0
    }
    
    # Lists to store cardinality properties
    card_k: List[int] = []
    card_size: List[int] = []
    
    queue = obj if isinstance(obj, list) else [obj]
    # Use a stack for traversal to avoid recursion limits
    stack = list(queue)
    
    while stack:
        item = stack.pop()
        
        if isinstance(item, Lit):
            counts["lit"] += 1
        elif isinstance(item, Not):
            counts["not"] += 1
            stack.append(item.term)
        elif isinstance(item, And):
            counts["and"] += 1
            stack.extend(item.terms)
        elif isinstance(item, Or):
            counts["or"] += 1
            stack.extend(item.terms)
        elif isinstance(item, Imp):
            counts["imp"] += 1
            stack.append(item.a)
            stack.append(item.b)
        elif isinstance(item, Iff):
            counts["iff"] += 1
            stack.append(item.a)
            stack.append(item.b)
        elif isinstance(item, Xor):
            counts["xor"] += 1
            stack.append(item.a)
            stack.append(item.b)
        elif isinstance(item, Cardinality):
            if isinstance(item, AtLeast): counts["at_least"] += 1
            elif isinstance(item, AtMost): counts["at_most"] += 1
            elif isinstance(item, Exactly): counts["exactly"] += 1
            
            card_k.append(item.k)
            card_size.append(len(item.vars))
            
    return ConstraintProfile(
        counts=counts,
        cardinalities={
            "all_k": sorted(card_k),
            "all_size": sorted(card_size)
        }
    )
