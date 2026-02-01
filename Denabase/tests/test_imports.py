import pytest
from pathlib import Path
from Denabase import DenaBase
from Denabase.core.types import sha256_bytes
from Denabase.core.serialization import atomic_write_text, read_text

def test_denabase_import():
    """Verify that we can import DenaBase."""
    assert DenaBase is not None

def test_submodule_imports():
    """Verify submodule imports."""
    import Denabase.core.types
    import Denabase.cnf.cnf_io
    import Denabase.graph.lcg
    import Denabase.embed.index
    import Denabase.db.denabase
    import Denabase.verify.verifier

def test_sha256_bytes():
    """Verify sha256_bytes utility."""
    data = b"hello denabase"
    expected = "3a6aeda0f4fb85cb9674101894050a1ef8c323b9834760fb39520d65628b1ab8"
    assert sha256_bytes(data) == expected

def test_atomic_write_text(tmp_path):
    """Verify atomic_write_text and read_text."""
    test_file = tmp_path / "test_atomic.txt"
    content = "Atomic content"
    atomic_write_text(test_file, content)
    
    assert test_file.exists()
    assert read_text(test_file) == content

def test_validate_cnf_valid():
    """Verify valid CNF validation."""
    from Denabase.core.types import validate_cnf
    cnf = [[1, 2], [-1, 3]]
    # Should not raise
    validate_cnf(cnf)

def test_validate_cnf_invalid():
    """Verify invalid CNF validation."""
    from Denabase.core.errors import ValidationError
    from Denabase.core.types import validate_cnf
    
    with pytest.raises(ValidationError):
        validate_cnf([[0]]) # 0 is not a valid literal in this context (should be != 0)
    
    with pytest.raises(ValidationError):
        validate_cnf([[]]) # Empty clause
