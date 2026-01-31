from typing import Dict, Any, List
from engine.state import AgentState, ModelingConstraint
from engine.connectivity.base import ConnectivityEncoder

class RankTreeConnectivity(ConnectivityEncoder):
    @property
    def name(self) -> str: return "rank_tree_connectivity"

    def required_kinds(self) -> Dict[str, Any]:
        return {
            "connected_8": {
                "parameters": {"cells": "List[str]", "adjacency": "8", "root": "Optional[str]"},
                "description": "Grid connectivity using Rank Tree (Stub)."
            }
        }

    def validate(self, constraint: ModelingConstraint, state: AgentState) -> None:
        p = constraint.parameters
        if "cells" not in p or not isinstance(p["cells"], list): raise ValueError("connected_8 requires 'cells' list")
        for c in p["cells"]:
             if str(c).lstrip('-~') not in state.sat_variables: raise ValueError(f"Var {c} not found")

    def compile(self, constraint: ModelingConstraint, state: AgentState) -> List[List[int]]:
        import re
        p = constraint.parameters
        cells = p["cells"]
        root_name = p.get("root")
        if not root_name and cells:
            root_name = cells[0]
            
        # 1. Parse coordinates to find neighbors
        # Pattern: pos_r_x_y_t or similar. We look for the last two digits.
        coord_map = {} # name -> (x, y)
        p_coord = re.compile(r".*?(\d+)_(\d+)(?:_\d+)?$")
        
        valid_cells = []
        for c in cells:
            m = p_coord.search(c)
            if m:
                x, y = map(int, m.groups())
                coord_map[c] = (x, y)
                valid_cells.append(c)
        
        if not valid_cells: return []
        
        # 2. Adjacency (8-connected)
        adj = {c: [] for c in valid_cells}
        for i, c1 in enumerate(valid_cells):
            x1, y1 = coord_map[c1]
            for j in range(i + 1, len(valid_cells)):
                c2 = valid_cells[j]
                x2, y2 = coord_map[c2]
                if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                    adj[c1].append(c2)
                    adj[c2].append(c1)
        
        # 3. Auxiliary Variables: Rank i has rank k
        # r_{i, k} for cell i, rank k in [0, N-1]
        n_cells = len(valid_cells)
        # We need n_cells * n_cells variables
        # Format: __rank_{cell_name}_{k}
        
        clauses = []
        
        # Get Literals from state
        def get_v(name): return state.var_manager.declare(name)
        
        rank_vars = {} # (cell, rank) -> vid
        for c in valid_cells:
            # Reserve a block for this cell's ranks
            # We use state.var_manager.reserve_block for efficiency
            block = state.var_manager.reserve_block(n_cells, prefix=f"rank_{c}")
            for k, vid in enumerate(block):
                rank_vars[(c, k)] = vid
        
        # Root logic
        root_vid = get_v(root_name)
        root_rank_0 = rank_vars[(root_name, 0)]
        
        # If root is active, it has rank 0
        clauses.append([-root_vid, root_rank_0])
        # If root has rank 0, it is active
        clauses.append([-root_rank_0, root_vid])
        
        # Other cells cannot have rank 0
        for c in valid_cells:
            if c == root_name: continue
            clauses.append([-rank_vars[(c, 0)]])
            
        # For each cell c, if c is active, it must have exactly one rank
        for c in valid_cells:
            x_i = get_v(c)
            c_ranks = [rank_vars[(c, k)] for k in range(n_cells)]
            
            # x_i => \/ r_{i, k}
            clauses.append([-x_i] + c_ranks)
            
            # r_{i, k} => x_i
            for r_ik in c_ranks:
                clauses.append([-r_ik, x_i])
                
            # At most one rank
            # (Simplified: pair-wise)
            for k1 in range(n_cells):
                for k2 in range(k1 + 1, n_cells):
                    clauses.append([-rank_vars[(c, k1)], -rank_vars[(c, k2)]])
                    
        # Connectivity: r_{i, k} => \/ r_{j, k-1} for j in neighbors
        for i, c in enumerate(valid_cells):
            if c == root_name: continue
            for k in range(1, n_cells):
                r_ik = rank_vars[(c, k)]
                neighbors = adj[c]
                if not neighbors:
                    # No neighbors -> cannot have rank k > 0
                    clauses.append([-r_ik])
                else:
                    predecessors = [rank_vars[(nc, k-1)] for nc in neighbors]
                    clauses.append([-r_ik] + predecessors)
                    
        return clauses
