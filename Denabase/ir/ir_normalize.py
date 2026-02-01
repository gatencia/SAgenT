from typing import Any, List, Set, Union
from Denabase.ir.ir_types import (
    BoolExpr, Lit, Not, And, Or, Imp, Iff, Xor
)

def build_stable_key(expr: BoolExpr) -> str:
    """Generates a stable key for structural comparison and sorting."""
    if isinstance(expr, Lit):
        return f"L:{'!' if expr.neg else ''}{expr.var.name}"
    elif isinstance(expr, Not):
        return f"N({build_stable_key(expr.term)})"
    elif isinstance(expr, (And, Or)):
        tag = "A" if isinstance(expr, And) else "O"
        sub_keys = sorted([build_stable_key(t) for t in expr.terms])
        return f"{tag}({','.join(sub_keys)})"
    elif isinstance(expr, Imp):
        return f"I({build_stable_key(expr.a)},{build_stable_key(expr.b)})"
    elif isinstance(expr, Iff):
        sub = sorted([build_stable_key(expr.a), build_stable_key(expr.b)])
        return f"F({','.join(sub)})"
    elif isinstance(expr, Xor):
        sub = sorted([build_stable_key(expr.a), build_stable_key(expr.b)])
        return f"X({','.join(sub)})"
    return "???"

def flatten(expr: BoolExpr) -> BoolExpr:
    """Flattens nested And/Or."""
    if isinstance(expr, And):
        flattened_terms = []
        for t in expr.terms:
            t_flat = flatten(t)
            if isinstance(t_flat, And):
                flattened_terms.extend(t_flat.terms)
            else:
                flattened_terms.append(t_flat)
        # Sort for determinism
        flattened_terms.sort(key=build_stable_key)
        return And(terms=flattened_terms)
    
    if isinstance(expr, Or):
        flattened_terms = []
        for t in expr.terms:
            t_flat = flatten(t)
            if isinstance(t_flat, Or):
                flattened_terms.extend(t_flat.terms)
            else:
                flattened_terms.append(t_flat)
        # Sort for determinism
        flattened_terms.sort(key=build_stable_key)
        return Or(terms=flattened_terms)
    
    if isinstance(expr, Not):
        return Not(term=flatten(expr.term))
    if isinstance(expr, Imp):
        return Imp(a=flatten(expr.a), b=flatten(expr.b))
    if isinstance(expr, Iff):
        return Iff(a=flatten(expr.a), b=flatten(expr.b))
    if isinstance(expr, Xor):
        return Xor(a=flatten(expr.a), b=flatten(expr.b))
    
    return expr

def rewrite_to_core(expr: BoolExpr) -> BoolExpr:
    """Rewrites Imp/Iff/Xor into And/Or/Not."""
    if isinstance(expr, Lit):
        return expr
    
    if isinstance(expr, Not):
        return Not(term=rewrite_to_core(expr.term))
    
    if isinstance(expr, And):
        return And(terms=[rewrite_to_core(t) for t in expr.terms])
    
    if isinstance(expr, Or):
        return Or(terms=[rewrite_to_core(t) for t in expr.terms])
    
    if isinstance(expr, Imp):
        # a -> b  =>  !a \/ b
        return Or(terms=[Not(term=rewrite_to_core(expr.a)), rewrite_to_core(expr.b)])
    
    if isinstance(expr, Iff):
        # a <-> b => (a /\ b) \/ (!a /\ !b)
        a_core = rewrite_to_core(expr.a)
        b_core = rewrite_to_core(expr.b)
        return Or(terms=[
            And(terms=[a_core, b_core]),
            And(terms=[Not(term=a_core), Not(term=b_core)])
        ])
    
    if isinstance(expr, Xor):
        # a ^ b   => (a /\ !b) \/ (!a /\ b)
        a_core = rewrite_to_core(expr.a)
        b_core = rewrite_to_core(expr.b)
        return Or(terms=[
            And(terms=[a_core, Not(term=b_core)]),
            And(terms=[Not(term=a_core), b_core])
        ])
    
    return expr

def push_not_inward(expr: BoolExpr) -> BoolExpr:
    """Pushes Not inward using DeMorgan and double negation removal."""
    if not isinstance(expr, Not):
        if isinstance(expr, And):
            return And(terms=[push_not_inward(t) for t in expr.terms])
        if isinstance(expr, Or):
            return Or(terms=[push_not_inward(t) for t in expr.terms])
        return expr
    
    # It's a Not(term)
    term = expr.term
    
    if isinstance(term, Not):
        # !!a => a
        return push_not_inward(term.term)
    
    if isinstance(term, And):
        # !(a /\ b) => !a \/ !b
        return Or(terms=[push_not_inward(Not(term=t)) for t in term.terms])
    
    if isinstance(term, Or):
        # !(a \/ b) => !a /\ !b
        return And(terms=[push_not_inward(Not(term=t)) for t in term.terms])
    
    if isinstance(term, Lit):
        # !(lit(v, neg)) => lit(v, !neg)
        return Lit(var=term.var, neg=not term.neg)
    
    # If we haven't rewritten Imp/Iff/Xor yet, we handle them as generic terms
    # But usually rewrite_to_core is called first.
    return expr

def normalize_ir(expr: BoolExpr) -> BoolExpr:
    """Full normalization pipeline: rewrite -> push not -> flatten."""
    expr = rewrite_to_core(expr)
    expr = push_not_inward(expr)
    expr = flatten(expr)
    return expr
