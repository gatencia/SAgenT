import time
from typing import Optional, List, Callable, Any, Dict, Union, Tuple
from pydantic import BaseModel, Field
from pysat.solvers import Solver
from Denabase.cnf.cnf_types import CnfDocument
from Denabase.core.errors import VerificationError
from Denabase.verify.metamorphic import MetamorphicSuite

class VerificationConfig(BaseModel):
    """Configuration for the verification process."""
    solver_name: str = "cadical153" # Modern, fast solver often included in pysat
    solver_calls_max: int = 1000
    seconds_max: float = 10.0
    num_metamorphic: int = 0
    check_simplify_equisat: bool = False
    
class VerificationResult(BaseModel):
    """Result of a verification run."""
    outcome: str # PASSED, FAILED, UNKNOWN
    is_satisfiable: Optional[bool] = None
    witness_valid: Optional[bool] = None
    checks_run: List[str] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    core_size: Optional[int] = None # For UNSAT cores

class CnfVerifier:
    def __init__(self, config: VerificationConfig):
        self.config = config

    def _solve(self, doc: CnfDocument) -> Tuple[bool, Optional[List[int]], Optional[int]]:
        """
        Solves CNF. Returns (is_sat, model, core_size).
        """
        with Solver(name=self.config.solver_name, bootstrap_with=doc.clauses) as solver:
            is_sat = solver.solve()
            model = solver.get_model() if is_sat else None
            return is_sat, model, None 

    def verify(self, 
               doc: CnfDocument, 
               decoder: Optional[Callable[[List[int]], Any]] = None,
               checker: Optional[Callable[[Any], bool]] = None) -> VerificationResult:
        """
        Verifies a CNF document.
        """
        result = VerificationResult(outcome="UNKNOWN")
        result.checks_run.append("solve")
        
        start_time = time.time()
        
        try:
            # Enforce timeout check
            if time.time() - start_time > self.config.seconds_max:
                 result.failures.append("Timeout exceeded seconds_max before primary solve")
                 result.outcome = "FAILED"
                 result.stats["duration"] = time.time() - start_time
                 return result

            is_sat, model, core_len = self._solve(doc)
            result.is_satisfiable = is_sat
            result.core_size = core_len
            
            if is_sat:
                if decoder and checker:
                    if time.time() - start_time > self.config.seconds_max:
                         result.failures.append("Timeout exceeded seconds_max before witness check")
                         result.outcome = "FAILED"
                         result.stats["duration"] = time.time() - start_time
                         return result

                    result.checks_run.append("witness_check")
                    try:
                        witness = decoder(model)
                        valid = checker(witness)
                        result.witness_valid = valid
                        if not valid:
                            result.failures.append("Witness checker returned False")
                    except Exception as e:
                        result.failures.append(f"Witness decoding/checking failed: {e}")
                        result.witness_valid = False
            
            # Metamorphic Checks
            if self.config.num_metamorphic > 0:
                result.checks_run.append(f"metamorphic_{self.config.num_metamorphic}")
                suite = MetamorphicSuite(doc)
                
                # We expect all variants to have SAME satisfiability
                for i, variant in enumerate(suite.generate_variants(self.config.num_metamorphic)):
                    if time.time() - start_time > self.config.seconds_max:
                         result.failures.append("Timeout exceeded seconds_max during metamorphic testing")
                         break # Break loop to fail gracefully

                    var_sat, _, _ = self._solve(variant)
                    if var_sat != is_sat:
                        result.failures.append(f"Metamorphic divergence at variant {i}: original={is_sat}, variant={var_sat}")
                        break
            
            # Final disposition
            if not result.failures:
                result.outcome = "PASSED"
            else:
                result.outcome = "FAILED"
                
        except Exception as e:
            result.failures.append(f"Solver error or timeout: {e}")
            result.outcome = "FAILED"
            
        result.stats["duration"] = time.time() - start_time
        return result
