import inspect
import pytest
from Denabase.Denabase.db.denabase import DenaBase

def test_denabase_methods_unique():
    """
    Ensure DenaBase class has exactly one definition for add_ir and add_cnf.
    Source checks are needed because Python runtime overwrites duplicates.
    """
    source = inspect.getsource(DenaBase)
    
    # We look for "def add_ir" lines
    add_ir_count = source.count("def add_ir")
    add_cnf_count = source.count("def add_cnf")
    
    assert add_ir_count == 1, f"Found {add_ir_count} definitions of add_ir"
    assert add_cnf_count == 1, f"Found {add_cnf_count} definitions of add_cnf"

def test_expected_public_api():
    """
    Ensure critical public methods exist.
    """
    public_methods = [
        "add_ir",
        "add_cnf",
        "query_similar",
        "rebuild_index",
        "scan_integrity"
    ]
    
    for method in public_methods:
        assert hasattr(DenaBase, method), f"DenaBase missing expected method {method}"
