import time
from typing import List, Dict, Any, Optional, Tuple, Set
from pysat.solvers import Solver

class DebugHarness:
    """
    Implements advanced SAT debugging techniques:
    - Selector-based Grouping
    - UNSAT Core Extraction
    - Delta Debugging (DDMin)
    - Model Enumeration & Locking
    """
    def __init__(self, solver_name='g3'):
        self.solver_name = solver_name
        self.clauses: List[List[int]] = []
        
        # Group Metadata: group_id -> {selector_lit, constraint_info}
        self.groups: Dict[str, Any] = {}
        self.selector_to_group: Dict[int, str] = {}
        
        # Internal state
        self.max_var_id = 0
        self.next_selector = 0

    def load_problem(self, cnf_clauses: List[List[int]], current_max_var: int):
        """Standard load without groups (legacy support)."""
        self.clauses = list(cnf_clauses)
        self.max_var_id = current_max_var
        
    def reset_instrumentation(self, max_var_id: int):
        """Clear groups and prepare for instrumented loading."""
        self.clauses = []
        self.groups = {}
        self.selector_to_group = {}
        self.max_var_id = max_var_id
        # Start selectors safely above max_var
        self.next_selector = max_var_id + 1000 

    def add_group(self, group_id: str, group_clauses: List[List[int]], info: Dict[str, Any]):
        r"""
        Add a group of clauses controlled by a unique selector variable 'a_i'.
        Transformation: C -> (C \/ ~a_i).
        If a_i is TRUE, C must be satisfied.
        If a_i is FALSE, C is disabled.
        """
        sel_lit = self.next_selector
        self.next_selector += 1
        
        self.groups[group_id] = {
            "selector": sel_lit,
            "info": info,
            "clause_count": len(group_clauses)
        }
        self.selector_to_group[sel_lit] = group_id
        
        # Instrument: Each clause becomes (c1 \/ c2 ... \/ ~sel)
        for cl in group_clauses:
            new_cl = list(cl)
            new_cl.append(-sel_lit)
            self.clauses.append(new_cl)

    def solve(self, assumptions: List[int] = None) -> Tuple[bool, Optional[List[int]]]:
        if assumptions is None: assumptions = []
        with Solver(name=self.solver_name, bootstrap_with=self.clauses) as s:
            sat = s.solve(assumptions=assumptions)
            if sat:
                return True, s.get_model()
            return False, None

    def get_core(self, assumptions: List[int]) -> List[str]:
        """
        Returns list of group_IDs responsible for UNSAT.
        Requires solving under assumptions first.
        """
        core_groups = []
        with Solver(name=self.solver_name, bootstrap_with=self.clauses) as s:
            if not s.solve(assumptions=assumptions):
                core_lits = s.get_core() or []
                for l in core_lits:
                    # Core returns the literals from assumptions.
                    # Our selectors were passed as Positive literals.
                    if l in self.selector_to_group:
                        core_groups.append(self.selector_to_group[l])
        
        return core_groups

    def ddmin(self, all_groups: List[str]) -> List[str]:
        """
        Delta Debugging to find a 1-minimal UNSAT subset of groups.
        Ref: Zeller '02.
        Input: List of group IDs that are currently UNSAT together.
        Output: A smaller subset that is still UNSAT.
        """
        
        def test(subset_groups):
            if not subset_groups: return False
            assumps = [self.groups[g]["selector"] for g in subset_groups]
            is_sat, _ = self.solve(assumptions=assumps)
            return not is_sat 

        # Iterative DDMin (Simplified)
        # Avoid recursion errors and handle large cores
        n = 2
        current_set = list(all_groups)
        
        while len(current_set) > 1:
            subsets = []
            chunk_size = max(1, len(current_set) // n)
            for i in range(0, len(current_set), chunk_size):
                subsets.append(current_set[i:i + chunk_size])
            
            found = False
            # 1. Check Subsets
            for sub in subsets:
                if test(sub):
                    current_set = sub
                    n = 2
                    found = True
                    break
            if found: continue
            
            # 2. Check Complements
            if n < len(current_set):
                for sub in subsets:
                    complement = [item for item in current_set if item not in sub]
                    if test(complement):
                        current_set = complement
                        n = max(n - 1, 2)
                        found = True
                        break
                if found: continue
            
            if n < len(current_set):
                n = min(n * 2, len(current_set))
            else:
                break
                
        return current_set

    def diagnose(self) -> Dict[str, Any]:
        """
        Main Routine:
        1. Enable all groups.
        2. Solve.
        3. If SAT -> Enumerate models (Case A).
        4. If UNSAT -> Extraction Core & Shrink (Case B).
        """
        t0 = time.time()
        
        # 1. All Selectors
        all_sels = [g["selector"] for g in self.groups.values()]
        
        # Solve
        with Solver(name=self.solver_name, bootstrap_with=self.clauses) as s:
            is_sat = s.solve(assumptions=all_sels)
            
            report = {
                "status": "SAT" if is_sat else "UNSAT",
                "time_solve": time.time() - t0,
                "total_vars": self.max_var_id,
                "total_clauses": len(self.clauses),
                "total_groups": len(self.groups)
            }
            
            if is_sat:
                # CASE A: SAT (Maybe wrong?)
                # We will handle enumeration in the Model Enumeration Block below
                pass 
                
            else:
                # CASE B: UNSAT
                # Get Core
                core_lits = s.get_core() or []
                core_gids = [self.selector_to_group[l] for l in core_lits if l in self.selector_to_group]
                report["unsat_core_size"] = len(core_gids)
                report["unsat_core_groups"] = core_gids
                
                # Shrink (DDMin)
                # Only run if core is not empty and not too huge (limit time)
                if core_gids and len(core_gids) > 1:
                    t_min = time.time()
                    min_core = self.ddmin(core_gids)
                    report["minimized_core"] = min_core
                    report["time_minimize"] = time.time() - t_min
                else:
                    report["minimized_core"] = core_gids
                    
                # Rich Info for Conflicts
                report["conflict_info"] = {gid: self.groups[gid]["info"] for gid in report["minimized_core"]}
                    
        # Model Enumeration (Iterative Blocking)
        if report["status"] == "SAT":
            models_found = []
            
            # Create a persistent solver for enumeration
            with Solver(name=self.solver_name, bootstrap_with=self.clauses) as s:
                # Limit K=5
                for _ in range(5):
                    # Solve with enabled groups
                    if not s.solve(assumptions=all_sels):
                        break
                    
                    m = s.get_model()
                    models_found.append(list(m)) # Store copy
                    
                    # Create blocking clause: NOT(current_assignment)
                    # We block based on the active literals in the model
                    # Standard blocking: (l1 AND l2 ...) -> False  <=>  (~l1 OR ~l2 ...)
                    blocking = [-l for l in m]
                    s.add_clause(blocking)
            
            report["models"] = models_found
            report["model_count"] = len(models_found)

        return report
