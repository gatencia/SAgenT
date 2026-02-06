from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.induce.motif_miner import MotifMiner, CandidateGadget
from Denabase.Denabase.verify.verifier import CnfVerifier, VerificationConfig
from Denabase.Denabase.cnf.cnf_io import load_cnf
from Denabase.Denabase.gadgets.gadget_registry import registry, LearnedGadget
from Denabase.Denabase.cnf.cnf_types import CnfDocument

logger = logging.getLogger(__name__)

class GadgetInducer:
    """Verifies candidates and registers them as valid gadgets."""
    
    def __init__(self, db: DenaBase):
        self.db = db
        self.miner = MotifMiner(db)
        # We need a verifier with a solver.
        self.verifier = CnfVerifier(VerificationConfig())

    def run_induction_pipeline(self, output_dir: Path) -> List[str]:
        """Runs mining, verification, and registration. Returns list of induced gadget names."""
        candidates = self.miner.mine()
        induced_names = []
        
        for cand in candidates:
            try:
                if self._verify_candidate(cand):
                    # Register
                    # Create name: Type_N_EntryShortHash
                    short_id = cand.entry_id[:8]
                    name = f"Induced_{cand.inferred_type}_{cand.params.get('n', 'X')}_{short_id}"
                    
                    lg = LearnedGadget(
                        name=name,
                        entry_id=cand.entry_id,
                        params=cand.params,
                        family="induced"
                    )
                    
                    registry.register_learned(lg)
                    induced_names.append(name)
                    logger.info(f"Induced and registered gadget: {name}")
            except Exception as e:
                logger.error(f"Failed to induce candidate {cand.entry_id}: {e}")
                
        # Save registry state
        registry.save_learned(output_dir)
        return induced_names

    def _verify_candidate(self, cand: CandidateGadget) -> bool:
        """Performs rigorous semantic verification of the candidate."""
        # Load CNF
        entry = self.db.store.get_entry_record(cand.entry_id)
        if not entry: return False
        
        cnf_path = self.db.root / entry.paths["cnf"]
        doc = load_cnf(cnf_path)
        
        # Dispatch to specific property checker
        if cand.inferred_type == "ExactlyOne":
            return self._verify_exactly_one(doc, cand.params.get("n", 0))
        elif cand.inferred_type == "AtMostOne":
            return self._verify_at_most_one(doc, cand.params.get("n", 0))
            
        return False

    def _verify_at_most_one(self, doc: CnfDocument, n: int, allow_zero: bool = True) -> bool:
        # Check: 
        # 1. SAT for all weight <= 1
        # 2. UNSAT for weight > 1 (checking weight 2 is usually sufficient for AMO)
        
        if n == 0: return False
        
        inputs = list(range(1, n + 1))
        
        # Test 1: Empty set (all 0)
        # If allow_zero (standard AMO), must be SAT.
        # If not (EO), we treat this separately in EO check or assume caller handles it.
        if allow_zero:
            assumptions = [-x for x in inputs]
            res = self.verifier.verify(doc, assumptions=assumptions)
            if res.outcome != "PASSED" or not res.is_satisfiable:
                return False # AMO(0) should be SAT

        # Test 2: Weight 1 -> SAT for each i
        for i in inputs:
            # Assumptions: i is True, others False
            asm = [i] + [-x for x in inputs if x != i]
            res = self.verifier.verify(doc, assumptions=asm)
            if res.outcome != "PASSED" or not res.is_satisfiable:
                return False

        # Test 3: Weight 2 -> UNSAT for pairs
        import itertools
        for i, j in itertools.combinations(inputs, 2):
            asm = [i, j] # Others don't matter? Or force others False? 
            # If we force i,j True, and it's UNSAT, then good.
            # If it's SAT, then bad.
            # Ideally we force others to False to test just these two, but pairwise constraints hold regardless of others usually.
            res = self.verifier.verify(doc, assumptions=asm)
            if res.outcome != "PASSED" or res.is_satisfiable:
                return False
                
        return True

    def _verify_exactly_one(self, doc: CnfDocument, n: int) -> bool:
        # 1. Inherits AMO properties (pairwise mutual exclusion), but MUST forbid 0.
        if not self._verify_at_most_one(doc, n, allow_zero=False):
            return False
            
        # 2. Add property: Weight 0 is UNSAT (AtLeastOne)
        inputs = list(range(1, n + 1))
        assumptions = [-x for x in inputs]
        res = self.verifier.verify(doc, assumptions=assumptions)
        
        # Expect UNSAT for EO(0)
        if res.outcome != "PASSED" or res.is_satisfiable:
            return False
            
        return True
