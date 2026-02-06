from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from Denabase.Denabase.verify.verifier import VerificationResult, VerificationConfig

class CandidateResult(BaseModel):
    """Result of searching a specific encoding/solver variant."""
    recipe_name: str
    encoding_recipe: Dict[str, Any] # Serialized recipe
    solver_config: Dict[str, Any] # Serialized solver config
    verification_result: Dict[str, Any] # Outcome, duration, etc.
    score: float = 0.0

    @property
    def is_correct(self) -> bool:
        return self.verification_result.get("outcome") == "PASSED"

def score_candidate(res: VerificationResult) -> float:
    """
    Scores a verification result. 
    Higher is better? Or lower (time)?
    Let's make Higher Better for 'Score'.
    But usually we want minimal time.
    Score = 1.0 / (1.0 + duration) if Passed.
    0 if Failed.
    """
    if res.outcome != "PASSED":
        return 0.0
    
    duration = res.stats.get("duration", 9999.0)
    # Prefer faster
    return 100.0 / (1.0 + duration)
