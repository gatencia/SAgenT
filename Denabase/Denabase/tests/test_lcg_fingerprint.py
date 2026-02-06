import pytest
import networkx as nx
from Denabase.Denabase.cnf.cnf_types import CnfDocument
from Denabase.Denabase.graph.lcg import build_lcg
from Denabase.Denabase.graph.wl_hash import compute_wl_hash
from Denabase.Denabase.graph.fingerprints import make_fingerprint, compute_fingerprint

def test_lcg_properties():
    doc = CnfDocument(num_vars=2, clauses=[[1, -2]])
    g = build_lcg(doc)
    
    # Nodes: C0, L1, L-2. Total 3.
    assert g.number_of_nodes() == 3
    assert g.has_edge("C0", "L1")
    assert g.has_edge("C0", "L-2")
    
    # Check attributes
    assert g.nodes["C0"]["kind"] == "clause"
    assert g.nodes["L1"]["kind"] == "lit"
    assert g.nodes["L1"]["sign"] == 1
    assert g.nodes["L-2"]["sign"] == -1

def test_wl_hash_consistency():
    doc = CnfDocument(num_vars=2, clauses=[[1, -2]])
    
    # Run directly
    g = build_lcg(doc)
    h1 = compute_wl_hash(g)
    h2 = compute_wl_hash(g)
    assert h1 == h2

    # Via fingerprint
    fp1 = compute_fingerprint(doc)
    fp2 = compute_fingerprint(doc)
    
    assert fp1.wl_hash == fp2.wl_hash
    assert fp1.signature_key == fp2.signature_key

def test_fingerprint_shape():
    doc = CnfDocument(num_vars=2, clauses=[[1, -2]])
    fp = compute_fingerprint(doc)
    
    # Profile vector (16) + Invariants (3) = 19
    assert len(fp.feature_vector) == 19
    
def test_renaming_invariance():
    # A: [1, 2], [1, -2]
    docA = CnfDocument(num_vars=2, clauses=[[1, 2], [1, -2]])
    
    # B: [3, 4], [3, -4] -> Isomorphic structure
    docB = CnfDocument(num_vars=4, clauses=[[3, 4], [3, -4]])
    
    fpA = compute_fingerprint(docA)
    fpB = compute_fingerprint(docB)
    
    assert fpA.wl_hash == fpB.wl_hash
