from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field
from Denabase.profile.profile_types import ConstraintProfile

class SolverConfig(BaseModel):
    """Configuration for a SAT solver run."""
    solver_name: str
    params: Dict[str, Union[str, int, float]] = Field(default_factory=dict)

class SolverSelector:
    """
    Selects a portfolio of solvers based on problem characteristics.
    """
    
    def select(self, profile: ConstraintProfile) -> List[SolverConfig]:
        """
        Returns a ranked list of solver configurations.
        """
        portfolio = []
        
        # Extract basic stats
        n_vars = 0
        n_clauses = 0
        
        if profile.cnf_stats:
            n_vars = profile.cnf_stats.get("n_vars", 0)
            n_clauses = profile.cnf_stats.get("n_clauses", 0)
        else:
            # Fallback to counts if stats not computed
            n_vars = profile.counts.get("vars", 0)
            n_clauses = profile.counts.get("clauses", 0)
            
        # Heuristics
        
        # 1. Large instances -> Modern CDCL (Cadical, Kissat style)
        if n_vars > 10000 or n_clauses > 50000:
            portfolio.append(SolverConfig(solver_name="cadical153"))
            portfolio.append(SolverConfig(solver_name="glucose4"))
        else:
            # Small/Medium -> Glucose3, Minisat22 often very fast on structured problems
            portfolio.append(SolverConfig(solver_name="glucose3"))
            portfolio.append(SolverConfig(solver_name="m22")) # Minisat22
            
            # Add Cadical as backup
            portfolio.append(SolverConfig(solver_name="cadical153"))
            
        # 2. Add specific configurations based on density?
        # E.g. if extremely dense, maybe a solver tuned for random-sat?
        # For now, keep it simple.
        
        return portfolio
