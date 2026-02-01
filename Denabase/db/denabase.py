from pathlib import Path
from typing import List, Optional
import uuid

from Denabase.cnf.cnf_io import load_cnf
from Denabase.cnf.cnf_stats import compute_cnf_stats
from Denabase.embed.feature_embedder import FeatureEmbedder
from Denabase.embed.index import VectorIndex
from Denabase.db.store import JSONStore
from Denabase.db.schema import DBEntry
from Denabase.core.types import CNFMetadata

class DenaBase:
    """The main entry point for interacting with a Denabase."""
    
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        self.store = JSONStore(self.root)
        
        # Load embedder if exists, else create new
        embedder_path = self.root / "embedder.pkl"
        if embedder_path.exists():
            self.embedder = FeatureEmbedder.load(embedder_path)
        else:
            self.embedder = FeatureEmbedder()
            
        self.index = VectorIndex()
        
        # Load existing entries into index
        self._sync_index()

    def _sync_index(self):
        """Syncs the vector index with the storage."""
        entries = self.store.load_entries()
        for entry in entries:
            # Ensure embedding is list of floats
            vec = entry.embedding
            self.index.add(vec, entry.id)

    def add_cnf(self, cnf_path: str, family: str, problem_id: str, meta: Optional[dict] = None) -> str:
        """Adds a CNF file to the Denabase."""
        path = Path(cnf_path)
        cnf = load_cnf(path)
        stats = compute_cnf_stats(cnf)
        
        # This will now work even if not fitted (uses fallback)
        embedding = self.embedder.embed(cnf)
        
        metadata = CNFMetadata(
            family=family,
            problem_id=problem_id,
            custom=meta or {}
        )
        
        entry_id = str(uuid.uuid4())
        # Copy file to db root for permanence (simplified for skeleton)
        dest_path = self.root / f"{entry_id}.cnf"
        try:
             dest_path.write_text(path.read_text())
        except Exception:
             # If read_text fails (binary?), try bytes
             dest_path.write_bytes(path.read_bytes())
        
        entry = DBEntry(
            id=entry_id,
            metadata=metadata,
            stats=stats, # Matches CNFStats schema now
            embedding=embedding.tolist(),
            cnf_path=str(dest_path)
        )
        
        self.store.save_entry(entry)
        self.index.add(embedding.tolist(), entry_id)
        
        return entry_id

    def query_similar(self, cnf_path: str, topk: int = 5) -> List[DBEntry]:
        """Queries for similar CNFs in the Denabase."""
        cnf = load_cnf(Path(cnf_path))
        embedding = self.embedder.embed(cnf)
        
        matches = self.index.query(embedding.tolist(), k=topk)
        
        results = []
        for entry_id, score in matches:
            entry = self.store.get_entry(entry_id)
            if entry:
                results.append(entry)
        
        return results
