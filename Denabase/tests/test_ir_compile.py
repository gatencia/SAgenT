import pytest
import itertools
from pysat.solvers import Solver
from Denabase.ir.ir_types import (
    VarRef, Lit, And, Or, Not, Imp, Iff, Xor, AtLeast, AtMost, Exactly
)
from Denabase.ir.ir_compile import compile_ir

def evaluate_bool(expr, assignment: dict[str, bool]) -> bool:
    """Brute force evaluation of BoolExpr."""
    if isinstance(expr, Lit):
        val = assignment[expr.var.name]
        return not val if expr.neg else val
    if isinstance(expr, Not):
        return not evaluate_bool(expr.term, assignment)
    if isinstance(expr, And):
        return all(evaluate_bool(t, assignment) for t in expr.terms)
    if isinstance(expr, Or):
        return any(evaluate_bool(t, assignment) for t in expr.terms)
    if isinstance(expr, Iff):
        return evaluate_bool(expr.a, assignment) == evaluate_bool(expr.b, assignment)
    raise ValueError(f"Unknown expr: {type(expr)}")

def evaluate_card(card, assignment: dict[str, bool]) -> bool:
    """Brute force evaluation of Cardinality."""
    count = sum(1 for v in card.vars if assignment[v.name])
    if isinstance(card, Exactly):
        return count == card.k
    if isinstance(card, AtMost):
        return count <= card.k
    if isinstance(card, AtLeast):
        return count >= card.k
    raise ValueError(f"Unknown card: {type(card)}")

def check_equivalence(obj, var_names: list[str]):
    """Checks equivalence between IR and compiled CNF."""
    cnf, varmap = compile_ir(obj)
    
    # Iterate all 2^n assignments
    for values in itertools.product([False, True], repeat=len(var_names)):
        assignment = dict(zip(var_names, values))
        
        # Expected from IR
        if isinstance(obj, (Exactly)):
            expected = evaluate_card(obj, assignment)
        else:
            expected = evaluate_bool(obj, assignment)
            
        # Actual from SAT
        with Solver(name="glucose4") as solver:
            for clause in cnf:
                solver.add_clause(clause)
            
            # Assumptions forcing base variables to match the assignment
            assumptions = []
            for name, val in assignment.items():
                vid = varmap[name]
                assumptions.append(vid if val else -vid)
            
            actual = solver.solve(assumptions=assumptions)
            
        assert actual == expected, f"Failed at {assignment} for {obj}"

def test_ir_equivalence_basic():
    a, b, c = VarRef(name="a"), VarRef(name="b"), VarRef(name="c")
    
    # (a OR b) AND (NOT a OR c)
    expr1 = And(terms=[
        Or(terms=[Lit(var=a), Lit(var=b)]),
        Or(terms=[Lit(var=a, neg=True), Lit(var=c)])
    ])
    check_equivalence(expr1, ["a", "b", "c"])

def test_ir_equivalence_iff():
    a, b = VarRef(name="a"), VarRef(name="b")
    expr2 = Iff(a=Lit(var=a), b=Lit(var=b))
    check_equivalence(expr2, ["a", "b"])

def test_ir_equivalence_exactly_one():
    a, b, c = VarRef(name="a"), VarRef(name="b"), VarRef(name="c")
    expr = Exactly(k=1, vars=[a, b, c])
    check_equivalence(expr, ["a", "b", "c"])

def test_ir_equivalence_at_most_two():
    a, b, c, d = [VarRef(name=n) for n in "abcd"]
    from Denabase.ir.ir_types import AtMost
    expr = AtMost(k=2, vars=[a, b, c, d])
    check_equivalence(expr, ["a", "b", "c", "d"])

def evaluate_generic(obj, assignment: dict[str, bool]) -> bool:
    if hasattr(obj, "k"):
        return evaluate_card(obj, assignment)
    return evaluate_bool(obj, assignment)

# Update check_equivalence to use evaluate_generic
def check_equivalence(obj, var_names: list[str]):
    cnf, varmap = compile_ir(obj)
    for values in itertools.product([False, True], repeat=len(var_names)):
        assignment = dict(zip(var_names, values))
        expected = evaluate_generic(obj, assignment)
        with Solver(name="glucose4") as solver:
            for clause in cnf:
                solver.add_clause(clause)
            assumptions = [varmap[n] if val else -varmap[n] for n, val in assignment.items()]
            actual = solver.solve(assumptions=assumptions)
        assert actual == expected, f"Failed at {assignment} for {obj}"
