import pytest
import numpy as np
from Denabase.cnf.cnf_types import CnfDocument
from Denabase.profile.cnf_profile import compute_cnf_profile
from Denabase.graph.fingerprints import compute_fingerprint
from Denabase.embed.feature_embedder import FeatureEmbedder
from Denabase.embed.index import VectorIndex
from Denabase.core.errors import EmbeddingError

def test_index_retrieval_flow(tmp_path):
    # 1. Create Document A
    docA = CnfDocument(num_vars=2, clauses=[[1, 2], [1, -2]])
    # We still need raw vectors to fit the embedder initially
    # Use internal helper or public compute_fingerprint for setup
    fpA = compute_fingerprint(docA, compute_cnf_profile(docA))
    
    # 2. Create Document B (Renamed vars: 1->3, 2->4)
    docB = CnfDocument(num_vars=4, clauses=[[3, 4], [3, -4]])
    
    # 3. Create Document C (Distinct)
    docC = CnfDocument(num_vars=5, clauses=[[1, 2, 3], [-1, -4, -5]])
    fpC = compute_fingerprint(docC, compute_cnf_profile(docC))
    
    # Vectors for fitting
    vecs = [fpA.feature_vector, fpC.feature_vector]
    ids = ["docA", "docC"]
    
    # 4. Embedder
    embedder = FeatureEmbedder(output_dim=4, random_state=42)
    
    # NOW: Embed should WORK without fit (using fallback)
    vecA_unfitted = embedder.embed(docA)
    assert isinstance(vecA_unfitted, np.ndarray)
    assert vecA_unfitted.shape == (4,)
    
    # Check determinism of fallback
    vecA_unfitted_2 = embedder.embed(docA)
    np.testing.assert_array_almost_equal(vecA_unfitted, vecA_unfitted_2)

    # 5. Fit
    embedder.fit(vecs)
    assert embedder.fitted
    
    # Now we can use embed() with fitted scale/pca
    vecA_emb = embedder.embed(docA, compute_cnf_profile(docA))
    
    assert isinstance(vecA_emb, np.ndarray)
    assert vecA_emb.ndim == 1
    assert vecA_emb.dtype == np.float32
    # assert vecA_emb != vecA_unfitted (likely different, but fallback is random proj, pca is pca)
    
    # Batch embed
    docs_to_index = [docA, docC]
    profs_to_index = [compute_cnf_profile(docA), compute_cnf_profile(docC)]
    
    reduced_vecs = embedder.batch_embed(docs_to_index, profs_to_index)
    
    assert isinstance(reduced_vecs, np.ndarray)
    assert reduced_vecs.ndim == 2
    assert reduced_vecs.dtype == np.float32
    assert reduced_vecs.shape[0] == 2
    
    # 6. Index
    idx = VectorIndex(metric='cosine')
    idx.add(reduced_vecs.tolist(), ids)
    
    # 7. Query with B
    # Embed B
    profB = compute_cnf_profile(docB)
    vecB_emb = embedder.embed(docB, profB)
    
    # Note: query uses cosine distance/sim.
    results = idx.query(vecB_emb.tolist(), k=2)
    
    # Expect docA to be top result (structurally identical)
    top_id, score = results[0]
    assert top_id == "docA"
    # PCA might rotate things, but A and B have same fingerprint -> same vector -> embeddings are identical
    # so score should be very high (1.0 or 0.0 distance)
    # Cosine SIMILARITY: 1.0 is max.
    # Check metric implementation in VectorIndex: if faiss inner product, it returns similarity.
    # If sklearn nearest neighbors: it usually returns distances.
    # But wrapper usually normalizes this. Let's assume similarity > 0.99
    # Actually wait, FeatureEmbedder normalizes? No. VectorIndex usually normalizes for cosine.
    # Let's trust previous test logic was > 0.99
    assert score > 0.99 
    
    # 8. Persistence
    idx_path = tmp_path / "index.pkl"
    idx.save(idx_path)
    
    idx_loaded = VectorIndex.load(idx_path)
    results_loaded = idx_loaded.query(vecB_emb.tolist(), k=2)
    
    assert results_loaded[0][0] == "docA"
    assert np.isclose(results_loaded[0][1], score)

def test_embedder_persistence(tmp_path):
    embedder = FeatureEmbedder(output_dim=2)
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.1, 2.1, 3.1]]
    embedder.fit(data)
    res = embedder.transform(data)
    
    model_path = tmp_path / "embedder.pkl"
    embedder.save(model_path)
    
    loaded = FeatureEmbedder.load(model_path)
    res_loaded = loaded.transform(data)
    
    np.testing.assert_array_almost_equal(res, res_loaded)
