import hashlib
import networkx as nx
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from Denabase.cnf.cnf_types import CnfDocument
from Denabase.profile.profile_types import ConstraintProfile, profile_vector
from Denabase.graph.lcg import build_lcg
from Denabase.graph.wl_hash import compute_wl_hash

class Fingerprint(BaseModel):
    """
    Rich structural fingerprint for retrieval and deduplication.
    """
    content_hash: str
    wl_hash: str
    invariants: Dict[str, Any]
    signature_key: str
    feature_vector: List[float]

def make_fingerprint(doc: CnfDocument, profile: Optional[ConstraintProfile] = None) -> Fingerprint:
    """
    Generates a fingerprint for a CNF document.
    """
    # 1. Content Hash (Exact match)
    c_hash = doc.content_hash()
    
    # 2. Graph Invariants & WL Hash
    graph = build_lcg(doc)
    wl = compute_wl_hash(graph)
    
    # Basic graph invariants
    invariants = {
        "order": graph.number_of_nodes(),
        "size": graph.number_of_edges(),
        # Density: 2|E| / (|V|(|V|-1))
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    }
    
    # 3. Signature Key (Fuzzy match / Bucketing)
    # Combine WL hash with rough stats (e.g. num_vars, num_clauses)
    # We use bucketed counts to allow for slight variations if needed, 
    # but for now let's use exact counts for the signature key base.
    # Actually, let's just use WL + invariants as the key component.
    sig_base = f"{wl}|{doc.num_vars}|{len(doc.clauses)}"
    signature_key = hashlib.sha256(sig_base.encode("utf-8")).hexdigest()
    
    # 4. Feature Vector (ML / Vector Search)
    # Combine profile vector (if available) + invariants
    vec = []
    
    if profile:
        vec.extend(profile_vector(profile))
    else:
        # Fallback padding if no profile provided (should rarely happen in pipeline)
        vec.extend([0.0] * 16) # Profile vector length is 16
        
    vec.append(float(invariants["order"]))
    vec.append(float(invariants["size"]))
    vec.append(float(invariants["density"]))
    
    return Fingerprint(
        content_hash=c_hash,
        wl_hash=wl,
        invariants=invariants,
        signature_key=signature_key,
        feature_vector=vec
    )

# Alias for backward compatibility
compute_fingerprint = make_fingerprint
