from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir.ir_types import IR
from Denabase.Denabase.cnf.cnf_io import load_cnf
from Denabase.Denabase.cnf.cnf_types import CnfDocument
from Denabase.Denabase.profile.cnf_profile import compute_ir_profile, compute_cnf_profile
from Denabase.Denabase.induce.motif_miner import MotifMiner
from Denabase.Denabase.selection.encoding_selector import EncodingSelector
from Denabase.Denabase.db.schema import ProvenanceRecord
from Denabase.Denabase.trace import EncodingTrace

class DenabaseTool:
    """
    Stable interface for external agents/tools to interact with Denabase.
    """
    def __init__(self, db_root: str):
        self.db = DenaBase(db_root)
        self.miner = MotifMiner(self.db)
        self.enc_selector = EncodingSelector()

    def add_dimacs(self, 
                   cnf_path: str, 
                   family: str, 
                   problem_id: str, 
                   meta: Dict[str, Any] = None,
                   verify: bool = False) -> str:
        """
        Ingests a DIMACS CNF file.
        Returns Entry ID.
        """
        doc = load_cnf(Path(cnf_path))
        return self.add_cnf(doc, family, problem_id, meta, verify=verify)

    def add_verified_dimacs(self, 
                            cnf_path: str, 
                            family: str, 
                            problem_id: str, 
                            meta: Dict[str, Any] = None) -> str:
        """
        Deprecated alias for add_dimacs.
        """
        import logging
        logging.warning("add_verified_dimacs is deprecated. Use add_dimacs(..., verify=True) instead.")
        return self.add_dimacs(cnf_path, family, problem_id, meta, verify=True)

    def add_cnf(self, 
                doc: CnfDocument, 
                family: str, 
                problem_id: str, 
                meta: Dict[str, Any] = None,
                verify: bool = False) -> str:
        """
        Ingests a CnfDocument object directly.
        """
        return self.db.add_cnf(doc, family, problem_id, meta, verify=verify)

    def add_verified_cnf(self, 
                         doc: CnfDocument, 
                         family: str, 
                         problem_id: str, 
                         meta: Dict[str, Any] = None) -> str:
        """
        Deprecated alias for add_cnf.
        """
        import logging
        logging.warning("add_verified_cnf is deprecated. Use add_cnf(..., verify=True) instead.")
        return self.add_cnf(doc, family, problem_id, meta, verify=True)

    def add_verified_ir(self, 
                        ir_obj: Any, # Dict or List (JSON-like)
                        family: str, 
                        problem_id: str, 
                        meta: Dict[str, Any] = None,
                        verify: bool = True) -> str:
        """
        Ingests a verified IR object (high-level constraints).
        Returns Entry ID.
        """
        # Validate/Deserialize if needed, but add_ir handles it
        # Actually add_ir expects Pydantic model or list ot models?
        # Let's ensure it's compatible. add_ir expects `Any`, calls normalize_ir.
        # normalize_ir handles dict/list -> Pydantic.
        return self.db.add_ir(ir_obj, family, problem_id, meta, verify=verify)

    def add_satbench_case(self,
                          dataset_id: str,
                          family: str,
                          problem_id: str,
                          nl_text: str,
                          expected_label: Any,
                          split: str = None,
                          cnf_path: str = None,
                          cnf_dimacs: str = None,
                          tags: List[str] = None,
                          user_meta: Dict[str, Any] = None,
                          verify: bool = False) -> str:
        """
        Ingests a SAT-Bench case with metadata and NL.
        """
        return self.db.add_satbench_case(
            dataset_id=dataset_id,
            family=family,
            problem_id=problem_id,
            nl_text=nl_text,
            expected_label=expected_label,
            split=split,
            cnf_path=cnf_path,
            cnf_dimacs=cnf_dimacs,
            tags=tags,
            user_meta=user_meta,
            verify=verify
        )

    def retrieve_similar(self, 
                         query_obj: Any, 
                         topk: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves similar entries for a given query (CNF or IR).
        """
        return self.db.query_similar(query_obj, topk=topk)

    def suggest_gadgets(self, 
                        query_obj: Any, 
                        topk: int = 5) -> Dict[str, Any]:
        """
        Suggests gadgets based on similar entries and motif mining.
        Returns: {
            "neighbors": [list of gadgets found in neighbors],
            "inferred": [list of gadgets inferred from query structure]
        }
        """
        # 1. Neighbors
        neighbors = self.retrieve_similar(query_obj, topk=topk)
        neighbor_gadgets = []
        for n in neighbors:
            eid = n["entry_id"]
            rec = self.db.store.get_entry_record(eid)
            # Try load provenance artifact
            # DenaBase doesn't expose retrieving artifact directly?
            # FileStore has get_artifact(eid, name).
            try:
                prov_path = self.db.store.get_artifact_path(eid, "provenance")
                if prov_path.exists():
                     # ProvenanceRecord
                     # Should be JSON loadable
                     with open(prov_path, "r") as f:
                         data = json.load(f)
                         # Pydantic model ProvenanceRecord
                         # data["gadgets"]
                         if "gadgets" in data:
                             neighbor_gadgets.extend(data["gadgets"])
            except Exception:
                pass # Provenance might not exist
        
        # Deduplicate
        neighbor_gadgets = list(set(neighbor_gadgets))
        
        # 2. Inferred (Motif Mining)
        # Note: MotifMiner scans DB entries usually.
        # Can we run it on an object?
        # MotifMiner methods rely on 'EntryRecord' iteration usually.
        # But we can perhaps extract profile/features and lookup known gadgets?
        # Wait, MotifMiner scans DB to FIND gadgets.
        # AutoDoc summarizes known ones.
        # Maybe we want to recognize gadgets IN the query?
        # Or suggest gadgets that *could* be used?
        # "suggest_gadgets" usually implies "what to use".
        # If I give IR with raw clauses, maybe it suggests "Use ExactlyOne"?
        # That's recognition.
        # Let's assume we return neighbor gadgets + maybe recognized patterns.
        # Since I configured miner to scan DB, I can't easily run it on transient obj.
        # I'll rely on neighbor gadgets for "suggestion".
        
        return {
            "neighbor_gadgets": neighbor_gadgets,
            "inferred": [] # Todo: implement on-the-fly recognition
        }

    def recommend_encoding_recipe(self, 
                                  ir_obj: Any) -> Dict[str, Any]:
        """
        Recommends an encoding recipe based on structure and neighbors.
        """
        # 1. Base Heuristic
        ir_prof = compute_ir_profile(ir_obj)
        recipe = self.enc_selector.select(ir_prof)
        
        # 2. Neighbor Telemetry (Placeholder)
        # neighbors = self.retrieve_similar(ir_obj, topk=5)
        # for n in neighbors:
        #    ... check their telemetry if available ...
        #    ... if 'pairwise' worked better, maybe suggestion adjustment?
        
        # For now return base recipe + note
        return {
            "recipe": recipe.model_dump(),
            "source": "heuristic", 
            "notes": recipe.notes
        }

    def attach_trace(self, entry_id: str, trace: Union[Dict[str, Any], EncodingTrace]) -> None:
        """
        Attaches a trace to an entry. trace can be a dict or EncodingTrace model.
        """
        if isinstance(trace, dict):
            trace = EncodingTrace(**trace)
        self.db.attach_trace(entry_id, trace)
