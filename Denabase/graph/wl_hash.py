import hashlib
import networkx as nx
from typing import List, Dict, Any

def compute_wl_hash(graph: nx.Graph, iterations: int = 2) -> str:
    """
    Computes a deterministic Weisfeiler-Lehman hash of the graph.
    """
    # 1. Initial Coloring
    # Hash of attributes: kind, sign, bucket
    colors: Dict[Any, str] = {}
    
    for node, attrs in graph.nodes(data=True):
        # Create a stable string representation of attributes
        # keys sorted ensures determinism
        attr_str = "|".join(f"{k}:{attrs[k]}" for k in sorted(attrs.keys()))
        colors[node] = hashlib.sha256(attr_str.encode("utf-8")).hexdigest()
        
    # 2. Refinement Loop
    for _ in range(iterations):
        new_colors = {}
        for node in graph.nodes():
            # Get neighbors' colors
            neighbor_colors = sorted([colors[n] for n in graph.neighbors(node)])
            
            # Combine own color + neighbor colors
            combined = colors[node] + "(" + ",".join(neighbor_colors) + ")"
            new_colors[node] = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        
        colors = new_colors
        
    # 3. Final Hash (Histogram of colors)
    final_colors = sorted(list(colors.values()))
    hist_str = ",".join(final_colors)
    return hashlib.sha256(hist_str.encode("utf-8")).hexdigest()
