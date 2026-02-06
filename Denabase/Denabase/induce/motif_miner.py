from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import logging

from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.profile.profile_types import ConstraintProfile

logger = logging.getLogger(__name__)

@dataclass
class CandidateGadget:
    entry_id: str
    inferred_type: str
    confidence: float
    params: Dict[str, Any]

class MotifMiner:
    """Mines the database for recurring motifs that match known gadget signatures."""
    
    def __init__(self, db: DenaBase):
        self.db = db

    def mine(self) -> List[CandidateGadget]:
        """Scans all valid entries to find gadget candidates."""
        candidates = []
        entries = self.db.store.load_entries() # Relies on our new load_entries
        
        for entry in entries:
            try:
                # Load profile
                prof_path = self.db.root / entry.paths["profile"]
                if not prof_path.exists(): continue
                
                with open(prof_path, "r") as f:
                    data = json.load(f)
                    prof = ConstraintProfile(**data)
                
                # Check for motifs
                
                # 1. ExactlyOne (k=1, size=N)
                # Signature: exactly_k=[1], count:exactly=1? Or just raw CNF stats?
                # If we used IR `Exactly` it shows up in profile `cardinalities["exactly_k"]`.
                # If we used raw CNF, we rely on `cnf_stats`.
                # Let's assume we are mining things previously ingested as IR or CNF that *look* like gadgets.
                
                # Heuristic: If it has exactly one "Exactly" constraint in profile card stats
                exact_ks = prof.cardinalities.get("exactly_k", [])
                if len(exact_ks) == 1 and exact_ks[0] == 1:
                    # Found an Exactly(1) candidate
                    # Get size (N)
                    sizes = prof.cardinalities.get("exactly_size", [])
                    n = sizes[0] if sizes else 0
                    
                    if n > 0:
                        candidates.append(CandidateGadget(
                            entry_id=entry.id,
                            inferred_type="ExactlyOne",
                            confidence=0.9,
                            params={"n": n}
                        ))
                        
                # 2. AtMostOne (k=1)
                am_ks = prof.cardinalities.get("at_most_k", [])
                if len(am_ks) == 1 and am_ks[0] == 1:
                    sizes = prof.cardinalities.get("at_most_size", [])
                    n = sizes[0] if sizes else 0
                    if n > 0:
                        candidates.append(CandidateGadget(
                            entry_id=entry.id,
                            inferred_type="AtMostOne",
                            confidence=0.8, # Slightly less confident without checking pairwise
                            params={"n": n}
                        ))

            except Exception as e:
                logger.warning(f"Failed to mine entry {entry.id}: {e}")
                continue
                
        return candidates
