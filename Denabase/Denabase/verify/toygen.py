import random
from typing import List
from Denabase.Denabase.cnf.cnf_types import CnfDocument

def generate_random_3sat(num_vars: int, num_clauses: int, seed: int = 42) -> CnfDocument:
    """
    Generates a random 3-SAT instance (k=3).
    """
    rng = random.Random(seed)
    clauses = []
    
    for _ in range(num_clauses):
        # Pick 3 distinct variables
        vars_in_clause = rng.sample(range(1, num_vars + 1), 3)
        clause = []
        for var in vars_in_clause:
            # Random sign
            sign = 1 if rng.choice([True, False]) else -1
            clause.append(var * sign)
        clauses.append(clause)
        
    return CnfDocument(num_vars=num_vars, clauses=clauses)

def generate_unsat_core() -> CnfDocument:
    """
    Generates a trivial UNSAT instance: (X) AND (NOT X).
    """
    return CnfDocument(num_vars=1, clauses=[[1], [-1]])
