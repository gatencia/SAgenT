class DenabaseError(Exception):
    """Base exception for all Denabase related errors."""
    pass

class ValidationError(DenabaseError):
    """Raised when data validation fails."""
    pass

class StorageError(DenabaseError):
    """Raised when database or file storage operations fail."""
    pass

class CNFError(DenabaseError):
    """Raised when there is an issue with CNF processing or parsing."""
    pass

class IRCompileError(DenabaseError):
    """Raised when compiling an Intermediate Representation to CNF fails."""
    pass

class VerificationError(DenabaseError):
    """Raised when verification of a SAT encoding or model fails."""
    pass

class EmbeddingError(DenabaseError):
    """Raised when feature extraction or embedding generation fails."""
    pass

class IndexError(DenabaseError):
    """Raised when vector index operations fail."""
    pass

class SelectionError(DenabaseError):
    """Raised when selecting components or gadgets fails."""
    pass
