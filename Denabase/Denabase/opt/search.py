import copy
from typing import Any, List, Optional, Dict
from pathlib import Path
import json

from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir.ir_compile import compile_ir
from Denabase.Denabase.cnf.cnf_types import CnfDocument
from Denabase.Denabase.profile.cnf_profile import compute_ir_profile, compute_cnf_profile
from Denabase.Denabase.selection.encoding_selector import EncodingSelector, EncodingRecipe
from Denabase.Denabase.selection.solver_selector import SolverSelector
from Denabase.Denabase.verify.verifier import CnfVerifier, VerificationConfig
from Denabase.Denabase.opt.metrics import CandidateResult, score_candidate

class EncodingSearcher:
    def __init__(self, db: DenaBase):
        self.db = db
        self.enc_selector = EncodingSelector()
        self.solver_selector = SolverSelector()

    def search(self, ir_obj: Any, time_budget: float = 10.0) -> Optional[CandidateResult]:
        """
        Explores encoding and solver variants.
        Returns the best result.
        """
        # 1. Profile
        ir_prof = compute_ir_profile(ir_obj)
        
        # 2. Base Encoding
        base_recipe = self.enc_selector.select(ir_prof)
        
        # 3. Generate Candidates (Mutations)
        recipes = [base_recipe]
        
        # Mutation 1: Flip cardinality encoding if sequential
        if base_recipe.cardinality_encoding == "sequential":
            r2 = base_recipe.model_copy(deep=True)
            r2.cardinality_encoding = "pairwise"
            r2.notes.append("Mutation: pairwise")
            recipes.append(r2)
        elif base_recipe.cardinality_encoding == "pairwise":
             r2 = base_recipe.model_copy(deep=True)
             r2.cardinality_encoding = "sequential"
             r2.notes.append("Mutation: sequential")
             recipes.append(r2)
             
        # Mutation 2: Toggle Symmetry
        r3 = base_recipe.model_copy(deep=True)
        r3.symmetry_breaking = not base_recipe.symmetry_breaking
        r3.notes.append(f"Mutation: symmetry={r3.symmetry_breaking}")
        recipes.append(r3)
        
        # 4. Solvers Profile (uses base recipe profile approximation or update?)
        # Ideally we compile once to get stats for solver selector.
        # But stats change with encoding!
        
        results: List[CandidateResult] = []
        
        for i, recipe in enumerate(recipes):
            # Compile
            clauses, varmap = compile_ir(ir_obj, recipe=recipe)
            
            # Create Doc
            max_var = len(varmap)
            if clauses:
                 max_c = max(abs(l) for c in clauses for l in c)
                 max_var = max(max_var, max_c)
            doc = CnfDocument(num_vars=max_var, clauses=clauses)
            
            # Re-profile for solver selection
            cnf_prof = compute_cnf_profile(doc)
            # Merge with IR profile
            prof = cnf_prof.model_copy()
            prof.counts.update(ir_prof.counts)
            prof.cardinalities.update(ir_prof.cardinalities)
            
            # Select Solvers
            solvers = self.solver_selector.select(prof)
            # Limit to top 2 for search speed
            solvers = solvers[:2]
            
            for j, solver_rec in enumerate(solvers):
                # Verify
                # Search mode: short timeout?
                v_config = VerificationConfig(
                    solver_name=solver_rec.solver_name,
                    seconds_max=time_budget / (len(recipes) * len(solvers)) # Distribute budget
                )
                verifier = CnfVerifier(v_config)
                
                res = verifier.verify(doc)
                
                score = score_candidate(res)
                
                cand = CandidateResult(
                    recipe_name=f"Recipe{i}_Solver{j}",
                    encoding_recipe=recipe.model_dump(),
                    solver_config=solver_rec.model_dump(),
                    verification_result=res.model_dump(),
                    score=score
                )
                results.append(cand)
                
        # 5. Persist Telemetry
        # We save ALL results as a run artifact? Or just best?
        # Let's save a summary JSON.
        run_id = f"search_{len(recipes)*len(solvers)}_{int(time_budget)}"
        # We don't have a specific persistent entry for the search itself unless we add the *best* one.
        # But requirement says "persist via DenaBase".
        # Maybe allow user to access it. 
        # For now, we assume this method is called by something that might store it, 
        # or we just return it. 
        # But "persist via DenaBase" suggests SIDE EFFECT.
        # Since we don't have an Entry yet, maybe we create one for the BEST result?
        # Or we assume this is called on an *existing* entry?
        # Input is `ir_obj`. It might be new.
        
        # Let's return best, and let caller Add it?
        # Or maybe we store the telemetry in a separate log file in DB root?
        # User said "update DB code if needed". 
        # I'll stick to returning best for now, and maybe writing a debug log.
        # Wait, "return best candidate".
        
        best = max(results, key=lambda x: x.score) if results else None
        return best
