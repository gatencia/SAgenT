from typing import List, Dict, Any, Tuple
from collections import Counter
import logging
import uuid
import math

from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.gadgets.macro_gadget import MacroGadget
from Denabase.Denabase.trace import EncodingTrace, TraceEvent

logger = logging.getLogger(__name__)

class StitchLiteMiner:
    """
    Mines recurring substructures from encoding traces to induce reusable macros.
    Implements a simplified version of Stitch (contigous n-grams + abstraction).
    """

    def __init__(self, db: DenaBase):
        self.db = db

    def mine(self, 
             min_freq: int = 2, 
             top_k: int = 5,
             max_ngram: int = 4) -> List[MacroGadget]:
        """
        Mines the database for frequent trace patterns and returns induced MacroGadgets.
        """
        # 1. Load all traces
        all_traces = self._load_all_traces()
        if not all_traces:
            logger.info("No traces found for induction.")
            return []

        # 2. Abstract traces to sequences of tokens
        # Token = (node_type, frozenset(params)?)
        # For v1: Token is just the node definition dict (type + params)
        # We need hashable tokens for n-gram counting.
        
        abstract_sequences = []
        for trace in all_traces:
            seq = []
            for event in trace.events:
                if event.kind == "IR_NODE":
                    # Create signature
                    # Ignore vars for now (vars are params), keep k/arity
                    sig = self._abstract_event(event.payload)
                    seq.append(sig)
            if seq:
                abstract_sequences.append(seq)

        # 3. Find Frequent N-grams
        counter = Counter()
        for seq in abstract_sequences:
            for n in range(2, max_ngram + 1):
                # Sliding window
                if len(seq) < n: continue
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i : i+n])
                    counter[ngram] += 1
        
        # 4. Score Candidates (Compression Gain)
        # Gain ~= (Freq - 1) * Size
        candidates = []
        for ngram, freq in counter.items():
            if freq < min_freq: continue
            
            # Simple size estimation
            size = len(ngram) # depth
            gain = (freq - 1) * size
            
            candidates.append({
                "ngram": ngram,
                "freq": freq,
                "gain": gain,
                "size": size
            })
            
        # Sort by gain desc
        candidates.sort(key=lambda x: x["gain"], reverse=True)
        top_cands = candidates[:top_k]
        
        # 5. Induce Macros
        macros = []
        for i, cand in enumerate(top_cands):
            macro = self._synthesize_macro(cand["ngram"], i)
            macros.append(macro)
            
        return macros

    def _load_all_traces(self) -> List[EncodingTrace]:
        traces = []
        for entry in self.db.store.load_entries():
            if entry.meta.has_trace:
                t = self.db.get_trace(entry.id)
                if t: traces.append(t)
        return traces

    def _abstract_event(self, payload: Dict[str, Any]) -> Tuple:
        """Creates a hashable signature for an IR node."""
        # Signature: (Type, k, arity)
        # We ignore specific variable names to allow generalization
        node_type = payload.get("type", "Unknown")
        k = payload.get("k", -1)
        arity = payload.get("arity", 0)
        
        return (node_type, k, arity)

    def _synthesize_macro(self, ngram: Tuple, idx: int) -> MacroGadget:
        """Converts an n-gram signature back into a MacroGadget template."""
        # Ngram is list of (Type, k, arity).
        # We need to reconstruct a useful template.
        # Since we lost variable flows in simple abstraction, we assume:
        # A sequence of constraints on disjoint sets of vars? 
        # Or shared vars?
        # For StitchLite v1: We assume they are independent or we create placeholders.
        # To make it compiled-able, we need to declare variables in the params.
        
        template = []
        params_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # We'll simple create a macro that applies these constraints
        # Parameters will be:
        # vars_0 for step 0
        # vars_1 for step 1
        # Unless we can infer sharing (e.g. if we mined variable names too).
        # Without variable flow mining, we can only propose "Composition of Constraints".
        
        for step_i, item in enumerate(ngram):
            node_type, k, arity = item
            
            step_vars_key = f"step_{step_i}_vars"
            
            node_def = {
                "type": node_type,
                "vars": f"${step_vars_key}" # Placeholder
            }
            if k != -1:
                node_def["k"] = k
                
            template.append(node_def)
            
            # Add to schema
            params_schema["properties"][step_vars_key] = {
                "type": "array", 
                "items": {"type": "string"},
                "minItems": arity if arity > 0 else 1
            }
            params_schema["required"].append(step_vars_key)

        return MacroGadget(
            name=f"auto_macro_{idx}_{uuid.uuid4().hex[:4]}",
            description=f"Auto-induced macro from frequent pattern of length {len(ngram)}",
            ir_template=template,
            params_schema=params_schema
        )
