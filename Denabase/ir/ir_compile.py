from typing import Dict, List, Set, Tuple, Union, Optional
import uuid
from Denabase.ir.ir_types import (
    BoolExpr, Lit, Not, And, Or, Imp, Iff, Xor, VarRef, Cardinality, AtLeast, AtMost, Exactly, FixedCNF
)
from Denabase.ir.ir_normalize import normalize_ir
from Denabase.cnf.cnf_types import CNFEncoding
from Denabase.trace import EncodingTrace, TraceEvent

class CompilationContext:
    def __init__(self, base_vars: List[str]):
        # Map variable names to IDs 1..n
        self.varmap = {name: i + 1 for i, name in enumerate(sorted(base_vars))}
        self.next_aux = len(self.varmap) + 1
        self.clauses: CNFEncoding = []

    def allocate_aux(self) -> int:
        aux = self.next_aux
        self.next_aux += 1
        return aux

    def add_clause(self, clause: List[int]):
        self.clauses.append(clause)

def tseitin(expr: BoolExpr, ctx: CompilationContext) -> int:
    """Tseitin transformation: returns the literal representing the expression."""
    if isinstance(expr, Lit):
        var_id = ctx.varmap[expr.var.name]
        return -var_id if expr.neg else var_id
    
    if isinstance(expr, Not):
        target = tseitin(expr.term, ctx)
        return -target
    
    if isinstance(expr, And):
        out = ctx.allocate_aux()
        inputs = [tseitin(t, ctx) for t in expr.terms]
        # out <-> (i1 /\ i2 /\ ...)
        # 1. out -> in1, out -> in2, ...
        # (-out \/ in)
        for i in inputs:
            ctx.add_clause([-out, i])
        # 2. (in1 /\ in2 /\ ...) -> out
        # (-in1 \/ -in2 \/ ... \/ out)
        ctx.add_clause([-i for i in inputs] + [out])
        return out
    
    if isinstance(expr, Or):
        out = ctx.allocate_aux()
        inputs = [tseitin(t, ctx) for t in expr.terms]
        # out <-> (i1 \/ i2 \/ ...)
        # 1. in1 -> out, in2 -> out, ...
        # (-in \/ out)
        for i in inputs:
            ctx.add_clause([-i, out])
        # 2. out -> (in1 \/ in2 \/ ...)
        # (-out \/ in1 \/ in2 \/ ...)
        ctx.add_clause([-out] + inputs)
        return out
    
    raise ValueError(f"Unsupported expression for Tseitin: {type(expr)}")

def encode_at_most(k: int, x: List[int], ctx: CompilationContext):
    """
    Encodes AtMost(k, x) using Sequential Counter.
    x is a list of literals.
    """
    n = len(x)
    if k >= n:
        return # Always satisfied
    if k == 0:
        # All literals must be false
        for lit in x:
            ctx.add_clause([-lit])
        return

    # s[i][j] means at least j of first i+1 variables were true
    s = {}
    for i in range(n - 1):
        for j in range(1, k + 1):
            s[(i, j)] = ctx.allocate_aux()

    # Base case: x[0] -> s[0][1]
    ctx.add_clause([-x[0], s[(0, 1)]])
    # x[0] -> !s[0][j] for j > 1 (handled by not creating s[0][j])

    for i in range(1, n - 1):
        # x[i] -> s[i][1]
        ctx.add_clause([-x[i], s[(i, 1)]])
        # s[i-1][1] -> s[i][1]
        ctx.add_clause([-s[(i-1, 1)], s[(i, 1)]])

        for j in range(2, k + 1):
            # x[i] /\ s[i-1][j-1] -> s[i][j]
            ctx.add_clause([-x[i], -s[(i-1, j-1)], s[(i, j)]])
            # s[i-1][j] -> s[i][j]
            ctx.add_clause([-s[(i-1, j)], s[(i, j)]])

        # Over-capacity check: x[i] /\ s[i-1][k] -> False
        ctx.add_clause([-x[i], -s[(i-1, k)]])

    # Final overflow check for the n-th variable
    ctx.add_clause([-x[n-1], -s[(n-2, k)]])

from itertools import combinations
from Denabase.selection.encoding_selector import EncodingRecipe

def encode_at_most_pairwise(x: List[int], ctx: CompilationContext):
    """
    Encodes AtMost(1, x) using Pairwise encoding.
    """
    for l1, l2 in combinations(x, 2):
        ctx.add_clause([-l1, -l2])

def encode_cardinality(card: Cardinality, ctx: CompilationContext, recipe: EncodingRecipe = None):
    """Encodes cardinality constraints."""
    x = [ctx.varmap[v.name] for v in card.vars]
    n = len(x)
    k = card.k
    
    # Defaults
    method = recipe.cardinality_encoding if recipe else "sequential"
    
    if isinstance(card, AtMost):
        if k == 1 and method == "pairwise":
            encode_at_most_pairwise(x, ctx)
        else:
            # Fallback to sequential for k > 1 or explicit sequential
            encode_at_most(k, x, ctx)
    elif isinstance(card, AtLeast):
        # AtLeast(k, x) is AtMost(n-k, !x)
        not_x = [-lit for lit in x]
        if n - k == 1 and method == "pairwise":
            encode_at_most_pairwise(not_x, ctx)
        else:
            encode_at_most(n - k, not_x, ctx)
    elif isinstance(card, Exactly):
        # Exactly(k, x) is AtMost(k, x) AND AtLeast(k, x)
        # Handle first part
        if k == 1 and method == "pairwise":
            encode_at_most_pairwise(x, ctx)
        else:
            encode_at_most(k, x, ctx)
            
        # Handle second part (AtLeast k) -> AtMost(n-k, !x)
        not_x = [-lit for lit in x]
        if n - k == 1 and method == "pairwise":
            encode_at_most_pairwise(not_x, ctx)
        else:
            encode_at_most(n - k, not_x, ctx)

def encode_fixed_cnf(obj: FixedCNF, ctx: CompilationContext):
    """
    Encodes a FixedCNF fragment by inlining clauses and remapping variables.
    obj.vars[i] (0-indexed) corresponds to internal variable i+1 (1-indexed).
    Internal variables > len(obj.vars) are treated as internal auxiliaries and mapped to new unique aux variables.
    """
    # 1. Build mapping table
    mapping = {}
    
    # Map ports (1..N)
    for i, v_ref in enumerate(obj.vars):
        internal_id = i + 1
        if v_ref.name not in ctx.varmap:
            raise ValueError(f"FixedCNF reference to unknown variable: {v_ref.name}")
        mapping[internal_id] = ctx.varmap[v_ref.name]
        
    # 2. Iterate clauses
    # We need to discover max internal var to know if there are hidden auxiliaries
    max_int_var = 0
    if obj.clauses:
        max_int_var = max(abs(l) for c in obj.clauses for l in c)
        
    num_ports = len(obj.vars)
    # Allocate new aux for internal auxiliaries (num_ports+1 ... max_int_var)
    for i in range(num_ports + 1, max_int_var + 1):
        mapping[i] = ctx.allocate_aux()
        
    # 3. Rewrite and Add
    for clause in obj.clauses:
        new_clause = []
        for lit in clause:
            var = abs(lit)
            is_neg = (lit < 0)
            
            if var not in mapping:
                # Should not happen if max_int_var calculation correct
                # Maybe 0? DIMACS shouldn't have 0.
                raise ValueError(f"FixedCNF contains unmapped variable {var}")
                
            mapped_var = mapping[var]
            new_clause.append(-mapped_var if is_neg else mapped_var)
            
        ctx.add_clause(new_clause)

def compile_ir(ir_obj: Union[BoolExpr, Cardinality, FixedCNF, List[Union[BoolExpr, Cardinality, FixedCNF]]], 
               recipe: EncodingRecipe = None,
               trace: Optional[EncodingTrace] = None) -> Tuple[CNFEncoding, Dict[str, int]]:
    """Compiles IR to CNF."""
    objs = ir_obj if isinstance(ir_obj, list) else [ir_obj]
    
    # Trace Logging (Pre-compilation)
    if trace is not None:
        for i, obj in enumerate(objs):
            payload = {"step_index": i}
            if isinstance(obj, (AtLeast, AtMost, Exactly)):
                payload["type"] = obj.__class__.__name__
                payload["k"] = obj.k
                payload["vars"] = [v.name for v in obj.vars]
                payload["arity"] = len(obj.vars)
                # Parameters for recipes could be logged here too
            elif isinstance(obj, (And, Or, Not, Imp, Iff, Xor)):
                payload["type"] = obj.__class__.__name__
                # Shallow log of structure
                if hasattr(obj, 'terms'):
                    payload["arity"] = len(obj.terms)
                else:
                    payload["arity"] = 2 # binop
            elif isinstance(obj, Lit):
                payload["type"] = "Lit"
                payload["var"] = obj.var.name
            elif isinstance(obj, FixedCNF):
                payload["type"] = "FixedCNF"
                payload["vars"] = [v.name for v in obj.vars]
                payload["arity"] = len(obj.vars)
            
            trace.events.append(TraceEvent(kind="IR_NODE", payload=payload))
    
    # Collect all base variable names
    base_vars: Set[str] = set()
    def collect_vars(e):
        if isinstance(e, Lit): base_vars.add(e.var.name)
        elif isinstance(e, Not): collect_vars(e.term)
        elif isinstance(e, (And, Or, Imp, Iff, Xor)):
            for t in (e.terms if hasattr(e, 'terms') else [e.a, e.b]):
                collect_vars(t)
        elif isinstance(e, (AtLeast, AtMost, Exactly, FixedCNF)):
            for v in e.vars: base_vars.add(v.name)
 
    for obj in objs:
        collect_vars(obj)
    
    ctx = CompilationContext(list(base_vars))
    
    for obj in objs:
        if isinstance(obj, (AtLeast, AtMost, Exactly)):
            encode_cardinality(obj, ctx, recipe=recipe)
        elif isinstance(obj, FixedCNF):
            encode_fixed_cnf(obj, ctx)
        else:
            # Normalize boolean expression
            norm_expr = normalize_ir(obj)
            root_lit = tseitin(norm_expr, ctx)
            ctx.add_clause([root_lit])
            
    # Deterministic clause ordering
    ctx.clauses.sort(key=lambda c: (len(c), sorted([abs(l) for l in c]), c))
    
    # Trace Logging (Post-compilation)
    if trace is not None:
        trace.events.append(TraceEvent(kind="CNF_EMIT", payload={
            "clauses": len(ctx.clauses),
            "vars": len(ctx.varmap),
            "aux_vars": ctx.next_aux - 1 - len(ctx.varmap)
        }))
    
    return ctx.clauses, ctx.varmap
