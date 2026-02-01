import random
import copy
from typing import List, Dict, Iterator
from Denabase.cnf.cnf_types import CnfDocument

def rename_vars(doc: CnfDocument, seed: int) -> CnfDocument:
    """Randomly permutes variable IDs."""
    if doc.num_vars == 0:
        return doc
        
    rng = random.Random(seed)
    # Create permutation 1..N -> 1..N
    perm = list(range(1, doc.num_vars + 1))
    rng.shuffle(perm)
    
    # Map old -> new (1-based index needs -1 for list access if we used array, but dict is easier)
    mapping = {old: new for old, new in zip(range(1, doc.num_vars + 1), perm)}
    
    new_clauses = []
    for clause in doc.clauses:
        new_clause = []
        for lit in clause:
            var = abs(lit)
            sign = 1 if lit > 0 else -1
            new_var = mapping[var]
            new_clause.append(new_var * sign)
        new_clauses.append(new_clause)
        
    return CnfDocument(num_vars=doc.num_vars, clauses=new_clauses)

def permute_clauses(doc: CnfDocument, seed: int) -> CnfDocument:
    """Randomly reorders clauses."""
    rng = random.Random(seed)
    new_clauses = list(doc.clauses)
    rng.shuffle(new_clauses)
    return CnfDocument(num_vars=doc.num_vars, clauses=new_clauses)

def permute_literals(doc: CnfDocument, seed: int) -> CnfDocument:
    """Randomly reorders literals within each clause."""
    rng = random.Random(seed)
    new_clauses = []
    for clause in doc.clauses:
        new_c = list(clause)
        rng.shuffle(new_c)
        new_clauses.append(new_c)
    return CnfDocument(num_vars=doc.num_vars, clauses=new_clauses)

class MetamorphicSuite:
    """
    Generates semantically equivalent variants of a CNF document.
    """
    def __init__(self, doc: CnfDocument, seed: int = 42):
        self.doc = doc
        self.seed = seed
        
    def generate_variants(self, count: int) -> Iterator[CnfDocument]:
        """Yields 'count' randomized variants."""
        rng = random.Random(self.seed)
        for i in range(count):
            # Apply a mix of transforms
            variant = self.doc
            
            # Chain transformations
            s_mix = rng.randint(0, 100000)
            
            # Simple policy: always permute clauses and lits, randomly rename vars
            variant = permute_clauses(variant, s_mix + 1)
            variant = permute_literals(variant, s_mix + 2)
            
            if rng.choice([True, False]):
                variant = rename_vars(variant, s_mix + 3)
                
            yield variant
