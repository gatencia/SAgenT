import os
import pytest
import numpy as np
from pathlib import Path
from Denabase.Denabase.alpha.alpha_model import AlphaModel, AlphaExample, extract_alpha_features
from Denabase.Denabase.profile.profile_types import ConstraintProfile
from Denabase.Denabase.db.denabase import DenaBase

def test_extract_alpha_features_nesting():
    """Verify mean_clause_len is extracted from nested dict."""
    prof = ConstraintProfile(
        counts={"lit": 10},
        cnf_stats={
            "n_vars": 10,
            "n_clauses": 5,
            "clause_len": {"min": 2, "mean": 2.5, "max": 3},
            "polarity_ratio": 0.5
        }
    )
    feats = extract_alpha_features(prof, family="test", has_nl=True)
    
    assert feats["n_vars"] == 10
    assert feats["n_clauses"] == 5
    assert feats["mean_clause_len"] == 2.5
    assert feats["has_nl"] == 1
    assert feats["family"] == "test"

def test_alpha_prediction_directionality(tmp_path):
    """
    Assert predicted alpha for Large >= predicted alpha for Small.
    (Structure should dominate more on large/dense).
    """
    model = AlphaModel()
    
    # Training Data
    # 1. Small profile -> alpha = 0.55 (heuristics: logic/puzzle etc)
    prof_small = ConstraintProfile(
        cnf_stats={
            "n_vars": 5, 
            "n_clauses": 10, 
            "clause_len": {"mean": 2.0},
            "polarity_ratio": 0.5
        }
    )
    # 2. Large profile -> alpha = 0.85
    prof_large = ConstraintProfile(
        cnf_stats={
            "n_vars": 1000, 
            "n_clauses": 5000, 
            "clause_len": {"mean": 5.0},
            "polarity_ratio": 0.5
        }
    )
    
    feats_small = extract_alpha_features(prof_small, family="logic", has_nl=True)
    feats_large = extract_alpha_features(prof_large, family="industrial", has_nl=True)
    
    examples = [
        AlphaExample(features=feats_small, alpha=0.55),
        AlphaExample(features=feats_large, alpha=0.85),
        # Add some noise/buffer
        AlphaExample(features=feats_small, alpha=0.56),
        AlphaExample(features=feats_large, alpha=0.84)
    ]
    
    model.fit(examples)
    
    # Prediction
    pred_small = model.predict_alpha(feats_small)
    pred_large = model.predict_alpha(feats_large)
    
    print(f"DEBUG: pred_small={pred_small}, pred_large={pred_large}")
    assert pred_large > pred_small

def test_rebuild_alpha_model_smoke(tmp_path):
    """
    Verify that rebuild_alpha_model produces alpha_model.joblib 
    once threshold satisfied.
    """
    # Set threshold via env
    os.environ["DENABASE_ALPHA_MIN_EXAMPLES"] = "3"
    
    db = DenaBase.create(str(tmp_path))
    
    # Need 3 entries
    # add_cnf calls update_indexes which adds profile
    from Denabase.Denabase.cnf.cnf_io import read_dimacs_from_string
    doc = read_dimacs_from_string("p cnf 3 1\n1 2 3 0")
    
    db.add_cnf(doc, family="test", problem_id="p1")
    db.add_cnf(doc, family="test", problem_id="p2")
    db.add_cnf(doc, family="test", problem_id="p3")
    
    db.rebuild_alpha_model()
    
    # Check if file exists
    model_path = tmp_path / "indexes" / "alpha_model.joblib"
    assert model_path.exists()
    
    # Check if DB says it is fitted
    assert db.alpha_model.is_fitted
