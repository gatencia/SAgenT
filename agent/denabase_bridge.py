import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

from Denabase.db.denabase import DenaBase
from Denabase.gadgets.gadget_pack import load_pack
from Denabase.gadgets.gadget_registry import registry
from Denabase.trace import EncodingTrace

logger = logging.getLogger(__name__)

class DenabaseBridge:
    """
    Bridge between the monolithic Agent and Denabase.
    Handles retrieval, gadget loading, and trace storage.
    """
    _instance = None

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db = DenaBase.open(self.db_path) if self.db_path.exists() else DenaBase.create(self.db_path)
        self._load_latest_pack()
        
    @classmethod
    def get_instance(cls, db_path: str = "denabase_db") -> 'DenabaseBridge':
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def reload_gadgets(self):
        """Reloads the latest gadget pack (useful after Sleep cycle)."""
        self._load_latest_pack()

    def _load_latest_pack(self):
        """Loads the latest gadget pack if available."""
        packs_dir = self.db.root / "gadgets" / "packs"
        if not packs_dir.exists():
            return
        
        # Simple string sort for timestamped versions
        versions = sorted([d.name for d in packs_dir.iterdir() if d.is_dir()])
        if not versions:
            return
            
        latest = versions[-1]
        logger.info(f"Loading latest gadget pack: {latest}")
        try:
            load_pack(packs_dir / latest, registry)
        except Exception as e:
            logger.error(f"Failed to load pack {latest}: {e}")

    def retrieve_priors(self, nl_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves relevant priors from Denabase.
        Returns:
            {
                "analogs": List[Dict],
                "suggested_macros": List[Dict],
                "suggested_motifs": List[Dict],
                "pitfalls": List[str]
            }
        """
        results = self.db.query_similar(None, nl_query_text=nl_text, topk=top_k)
        
        analogs = []
        ir_counts = {}
        
        # 1. Process Analogs & Mine Traces
        for res in results:
            analogs.append({
                "problem_id": res["problem_id"],
                "family": res["family"],
                "score": res["final_score"],
                "summary": res.get("meta", {}).get("summary", "")
            })
            
            # Fetch Trace
            try:
                trace = self.db.get_trace(res["entry_id"])
                if trace:
                    for evt in trace.events:
                        if evt.kind == "IR_NODE":
                            # Extract normalized type if available, else usage raw kind
                            t = evt.payload.get("type", evt.payload.get("kind", "unknown"))
                            ir_counts[t] = ir_counts.get(t, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to mine trace for {res['entry_id']}: {e}")

        # 2. Identify Frequent Motifs
        suggested_motifs = [{"type": k, "count": v} for k, v in ir_counts.items()]
        suggested_motifs.sort(key=lambda x: x["count"], reverse=True)
        suggested_motifs = suggested_motifs[:10]

        # 3. Rank Macros
        # Score = Base (1.0) + Boost (IR frequency overlap)
        scored_macros = []
        
        # Helper: Get primitives used by a macro
        def get_macro_primitives(m_name: str) -> List[str]:
            gadget = registry.get(m_name)
            if not gadget: return []
            prims = set()
            # If it has ir_template (list of dicts)
            if hasattr(gadget, "ir_template"):
                for node in gadget.ir_template:
                    if "type" in node: prims.add(node["type"])
            return list(prims)

        for m_name in registry.list_gadgets():
            # Skip primitives themselves if they are in registry (usually registry contains only macros/adapters)
            # Check basic filtering
            if m_name in ["clause", "implies", "at_most_k", "at_least_k", "exactly_k", "exactly_one", "linear_leq", "linear_eq"]:
                continue
                
            score = 1.0 # Base score for existence
            reason_parts = []
            
            prims = get_macro_primitives(m_name)
            for p in prims:
                if p in ir_counts:
                    # Simple log-ish boost? Or linear?
                    # Let's do linear scaled down to avoid domination by one massive constraint?
                    # Or just +1 per occurrence type?
                    # Let's say +0.1 * occurrences 
                    val = ir_counts[p]
                    boost = min(5.0, val * 0.1) # Cap boost per primitive
                    score += boost
                    reason_parts.append(f"{p}({val})")
            
            if reason_parts:
                reason = f"Matches analog motifs: {', '.join(reason_parts)}"
            else:
                reason = "Available in registry"
                
            scored_macros.append({
                "name": m_name,
                "score": round(score, 2),
                "reason": reason
            })
            
        scored_macros.sort(key=lambda x: x["score"], reverse=True)
        
        # Rename old key "suggested_gadgets" to "suggested_macros" in return dict? 
        # User requested specific output structure.
        return {
            "analogs": analogs,
            "suggested_macros": scored_macros[:10],
            "suggested_motifs": suggested_motifs,
            "pitfall_checklist": [] 
        }
        
    def attach_trace(self, entry_id: str, trace: EncodingTrace):
        """Attach a trace to an entry."""
        self.db.attach_trace(entry_id, trace)
        
    def create_solution_entry(self, 
                              family: str, 
                              problem_id: str, 
                              cnf_clauses: List[List[int]], 
                              meta: Dict[str, Any]) -> str:
        """
        Creates a new entry for a solved problem.
        NOTE: This might be redundant if we want to store the IR instead of CNF.
        But the agent produces IR constraints -> CNF.
        Best to store the IR if possible, but the agent's 'IR' is just constraints list.
        For now, we store CNF + Meta.
        """
        from Denabase.cnf.cnf_types import CnfDocument
        
        # Calculate num_vars
        max_var = 0
        if cnf_clauses:
            max_var = max(abs(l) for c in cnf_clauses for l in c)
            
        doc = CnfDocument(clauses=cnf_clauses, num_vars=max_var)
        # Use add_cnf
        return self.db.add_cnf(doc, family=family, problem_id=problem_id, meta=meta, verify=False)
