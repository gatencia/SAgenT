import random
from typing import List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime, timezone
import copy

from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.induce.stitchlite import StitchLiteMiner
from Denabase.Denabase.gadgets.macro_gadget import MacroGadget
from Denabase.Denabase.gadgets.gadget_registry import registry
from Denabase.Denabase.verify.verifier import CnfVerifier, VerificationConfig, VerificationResult
from Denabase.Denabase.ir import compile_ir
from Denabase.Denabase.cnf.cnf_types import CnfDocument
from Denabase.Denabase.autodoc.autodoc import AutoDoc
from Denabase.Denabase.gadgets.gadget_pack import save_pack

logger = logging.getLogger(__name__)

class SleepRunner:
    """
    Orchestrates the sleep phase: Mining -> Induction -> Verification -> Packing.
    """
    def __init__(self, db: DenaBase):
        self.db = db
        self.miner = StitchLiteMiner(db)
        self.autodoc = AutoDoc()
        
    def run_sleep_cycle(self, 
                        min_freq: int = 2, 
                        top_k: int = 5, 
                        seed: int = 42) -> Dict[str, Any]:
        """
        Runs a complete sleep cycle.
        """
        random.seed(seed)
        start_time = datetime.now(timezone.utc)
        
        # 1. Mine Candidates
        logger.info("Mining candidates...")
        candidates = self.miner.mine(min_freq=min_freq, top_k=top_k)
        logger.info(f"Found {len(candidates)} candidates.")
        
        verified_macros = []
        verification_reports = {} # gadget_name -> report dict
        
        # 2. Verify Each Candidate
        for macro in candidates:
            report = self._verify_macro(macro)
            verification_reports[macro.name] = report.model_dump()
            
            if report.outcome == "PASSED":
                # 3. AutoDoc (Improve description)
                # We can try to summarize the template structure
                # For now just keep the induced description or append specifics?
                # Let's keep it simple.
                verified_macros.append(macro)
            else:
                logger.warning(f"Macro {macro.name} failed verification: {report.failures}")

        # 4. Register Verified Macros
        # Clear/Reset registry for packing? No, we pack what we found + existing?
        # User goal: "Save a versioned gadget pack".
        # Usually implies packing the NEWLY found ones, or the WHOLE library?
        # "Library learning" usually accumulates.
        # But if we want deterministic packing of this run, separate registry or filtering?
        # We'll register them into the GLOBAL registry for now.
        
        for m in verified_macros:
            registry.register_macro(m)
            
        # 5. Save Pack
        logger.info("Saving gadget pack...")
        pack_version = save_pack(self.db.root, registry, verification_reports)
        
        return {
            "version": pack_version,
            "candidates_found": len(candidates),
            "verified_count": len(verified_macros),
            "reports": verification_reports
        }

    def _verify_macro(self, macro: MacroGadget) -> VerificationResult:
        """
        Verifies a single macro by instantiating it with small parameters.
        """
        # Configuration for verification
        config = VerificationConfig(
            seconds_max=5.0,
            num_metamorphic=5, # Run permutation checks
            check_simplify_equisat=False
        )
        verifier = CnfVerifier(config)
        
        # Instantiate with toy parameters
        # Macro params schema typically: step_0_vars, step_1_vars...
        # We need to generate valid inputs.
        # Simple heuristic:
        # Check params_schema.
        # Generate var names "x0", "x1"...
        
        params = {}
        var_counter = 0
        
        try:
            props = macro.params_schema.get("properties", {})
            for p_name, p_def in props.items():
                if p_def.get("type") == "array" and p_def.get("items", {}).get("type") == "string":
                    # Generate variable list
                    # Size? minItems or default small (e.g. 3)
                    size = p_def.get("minItems", 3)
                    vars_list = [f"v{var_counter + i}" for i in range(size)]
                    var_counter += size
                    params[p_name] = vars_list
            
            # Build IR
            ir_out = macro.build_ir(params)
            
            # Compile
            clauses, varmap = compile_ir(ir_out)
            
            # Create Doc
            max_var = len(varmap)
            if clauses:
                max_c = max(abs(l) for c in clauses for l in c)
                max_var = max(max_var, max_c)
            doc = CnfDocument(num_vars=max_var, clauses=clauses)
            
            # Verify
            # We assume the macro should be Satisfiable for AT LEAST SOME inputs?
            # Actually, "Exactly(1, ...)" is SAT.
            # "AtMost(1, ...) AND AtLeast(2, ...)" is UNSAT.
            # If a macro is inherently UNSAT, is it a bad gadget?
            # Not necessarily, but usually gadgets are building blocks.
            # For now, we perform sanity checks. If metamorphic fails, it's bad.
            # If it passes metamorphic (invariance), it's likely a stable construct.
            
            result = verifier.verify(doc)
            
            # Check for Always-UNSAT
            # We enforce that a gadget must be satisfiable for at least the sampled parameters,
            # unless explicitly marked as a "contradiction gadget".
            if result.outcome == "PASSED":
                if result.is_satisfiable is False:
                    allow = macro.meta.get("allow_unsat", False)
                    if not allow:
                        result.outcome = "FAILED"
                        result.failures.append("Gadget is always UNSAT (and allow_unsat=False)")
            
            return result
            
        except Exception as e:
            # Conversion failed
            res = VerificationResult(outcome="FAILED")
            res.failures.append(f"Instantiation failed: {e}")
            return res
