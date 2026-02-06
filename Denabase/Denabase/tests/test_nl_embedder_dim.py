import os
import numpy as np
from pathlib import Path
from Denabase.Denabase.nl.nl_embedder import NLEmbedder

def test_nl_embedder_dim_small_corpus(tmp_path):
    """
    Test Case A: Fit on 3 short texts (small vocab) 
    and assert transform returns shape (n, 128) by default.
    """
    embedder = NLEmbedder(dim=128)
    texts = [
        "Exactly one person is guilty.",
        "Only one suspect did it.",
        "A logic puzzle about crime."
    ]
    embedder.fit(texts)
    
    assert embedder.is_fitted
    assert embedder.effective_dim < 128
    assert embedder.effective_dim >= 1
    
    vecs = embedder.transform(texts)
    assert vecs.shape == (3, 128)
    # Check that it's not all zeros (at least some components should be non-zero)
    assert np.any(vecs != 0)
    # Ensure rows are normalized (of length 1.0)
    norms = np.linalg.norm(vecs, axis=1)
    for n in norms:
        assert np.isclose(n, 1.0)

def test_nl_embedder_dim_empty_inputs(tmp_path):
    """
    Test Case B: Transform on ["", None, "hi"] returns shape (3, 128) 
    and first two are all zeros.
    """
    embedder = NLEmbedder(dim=128)
    texts_fit = ["hello world", "foo bar", "testing fit"]
    embedder.fit(texts_fit)
    
    query_texts = ["", None, "hi"]
    vecs = embedder.transform(query_texts)
    
    assert vecs.shape == (3, 128)
    # vecs[0] ("") should be zeros
    assert np.all(vecs[0] == 0)
    # vecs[1] (None) should be zeros
    assert np.all(vecs[1] == 0)
    # vecs[2] ("hi") might be zeros if "hi" is not in vocab, or non-zero if it is.
    # But shape must be (128,)
    assert vecs[2].shape == (128,)

def test_nl_embedder_save_load_dim(tmp_path):
    """
    Test Case C: Save/load preserves shape guarantee.
    """
    model_path = tmp_path / "embedder.joblib"
    embedder = NLEmbedder(dim=64)
    texts = ["apple banana", "orange grape", "fruit salad"]
    embedder.fit(texts)
    
    orig_vecs = embedder.transform(["apple"])
    assert orig_vecs.shape == (1, 64)
    
    embedder.save(model_path)
    
    new_embedder = NLEmbedder.load(model_path)
    assert new_embedder.dim == 64
    assert new_embedder.is_fitted == embedder.is_fitted
    assert new_embedder.effective_dim == embedder.effective_dim
    
    new_vecs = new_embedder.transform(["apple"])
    assert new_vecs.shape == (1, 64)
    assert np.allclose(orig_vecs, new_vecs)

def test_nl_embedder_unfitted_dim():
    """
    Ensure unfitted embedder returns zeros of correct shape.
    """
    embedder = NLEmbedder(dim=128)
    vecs = embedder.transform(["something"])
    assert vecs.shape == (1, 128)
    assert np.all(vecs == 0)
