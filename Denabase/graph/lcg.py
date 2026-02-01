import networkx as nx
from collections import defaultdict
from typing import Dict, Any, List
from Denabase.cnf.cnf_types import CnfDocument

def build_lcg(doc: CnfDocument) -> nx.Graph:
    """
    Builds a Literal-Clause Graph (LCG) from a CNF document.
    
    Nodes:
    - Clause nodes: "C{i}"
    - Literal nodes: "L{lit}"
    
    Edges:
    - (C{i}, L{lit}) if lit is in clause i
    
    Attributes:
    - Clause: kind="clause", sign=0
    - Lit: kind="lit", sign=sgn(lit), bucket=occurrence_count
    """
    G = nx.Graph()
    
    # Compute occurrence counts for bucketing
    counts = defaultdict(int)
    for clause in doc.clauses:
        for lit in clause:
            counts[lit] += 1
            
    # Add clauses and edges
    for i, clause in enumerate(doc.clauses):
        c_node = f"C{i}"
        G.add_node(c_node, kind="clause", sign=0, bucket=0)
        
        for lit in clause:
            l_node = f"L{lit}"
            # Add literal node if not exists
            if not G.has_node(l_node):
                count = counts[lit]
                # Simple bucket: log-ish scale or just raw count? 
                # Requirement says "binned var bucket". Let's use raw count for now as it's deterministic.
                # To make it more invariant to problem size, maybe relative? 
                # But for exact lookup/isomorphism, raw count is fine.
                G.add_node(l_node, kind="lit", sign=1 if lit > 0 else -1, bucket=count)
            
            G.add_edge(c_node, l_node)
            
    return G
