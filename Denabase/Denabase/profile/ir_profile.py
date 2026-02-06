from typing import Any, Dict, List
from Denabase.Denabase.profile.profile_types import ConstraintProfile
from Denabase.Denabase.ir.ir_types import (
    IR, Exactly, AtMost, AtLeast, 
    And, Or, Not, Lit, Imp, Iff, Xor
)

def compute_ir_profile(ir_obj: Any) -> ConstraintProfile:
    """
    Computes a profile from an IR object (high-level stats) using valid traversal.
    """
    if ir_obj is None:
        return ConstraintProfile(
            counts={"num_and": 0, "num_or": 0, "num_not": 0, "num_lit": 0},
            cardinalities={}
        )

    # Init counters
    counts = {
        "num_and": 0,
        "num_or": 0,
        "num_not": 0,
        "num_lit": 0,
        "num_imp": 0,
        "num_iff": 0, 
        "num_xor": 0,
        "num_card_exactly": 0,
        "num_card_atmost": 0,
        "num_card_atleast": 0
    }
    
    cardinalities = {
        "exactly_k": [], "exactly_size": [],
        "at_most_k": [], "at_most_size": [],
        "at_least_k": [], "at_least_size": []
    }
    
    def visit(node: Any):
        if isinstance(node, list):
            for x in node: visit(x)
            return

        # Explicit type checking based on IR types
        if isinstance(node, Lit):
            counts["num_lit"] += 1
            # Lit is terminal
            
        elif isinstance(node, Not):
            counts["num_not"] += 1
            visit(node.term)
            
        elif isinstance(node, And):
            counts["num_and"] += 1
            if hasattr(node, "terms"):
                for t in node.terms:
                    visit(t)
                    
        elif isinstance(node, Or):
            counts["num_or"] += 1
            if hasattr(node, "terms"):
                for t in node.terms:
                    visit(t)
                    
        elif isinstance(node, Imp):
            counts["num_imp"] += 1
            visit(node.a)
            visit(node.b)
            
        elif isinstance(node, Iff):
            counts["num_iff"] += 1
            visit(node.a)
            visit(node.b)
            
        elif isinstance(node, Xor):
            counts["num_xor"] += 1
            visit(node.a)
            visit(node.b)
            
        elif isinstance(node, Exactly):
            counts["num_card_exactly"] += 1
            cardinalities["exactly_k"].append(node.k)
            cardinalities["exactly_size"].append(len(node.vars))
            
        elif isinstance(node, AtMost):
            counts["num_card_atmost"] += 1
            cardinalities["at_most_k"].append(node.k)
            cardinalities["at_most_size"].append(len(node.vars))
            
        elif isinstance(node, AtLeast):
            counts["num_card_atleast"] += 1
            cardinalities["at_least_k"].append(node.k)
            cardinalities["at_least_size"].append(len(node.vars))
            
        # Fallback for unexpected types? Ignore.

    visit(ir_obj)
    
    # Aggregate lists for general heuristics
    cardinalities["all_k"] = (
        cardinalities["exactly_k"] + 
        cardinalities["at_most_k"] + 
        cardinalities["at_least_k"]
    )
    cardinalities["all_size"] = (
        cardinalities["exactly_size"] + 
        cardinalities["at_most_size"] + 
        cardinalities["at_least_size"]
    )
    
    # Also populate flattened list of group sizes and k values as requested
    cardinalities["card_group_sizes"] = cardinalities["all_size"]
    cardinalities["card_k_values"] = cardinalities["all_k"]

    return ConstraintProfile(
        counts=counts,
        cardinalities=cardinalities
    )
