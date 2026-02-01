import json
from pathlib import Path
from typing import Dict, List, Optional
from Denabase.db.schema import DBEntry

class JSONStore:
    """Simple JSON-based store for Denabase entries."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.entries_file = db_path / "entries.json"
        
        if not self.entries_file.exists():
            with open(self.entries_file, "w") as f:
                json.dump({}, f)

    def save_entry(self, entry: DBEntry):
        """Saves an entry to the store."""
        with open(self.entries_file, "r") as f:
            data = json.load(f)
        
        data[entry.id] = entry.model_dump()
        
        with open(self.entries_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_entries(self) -> List[DBEntry]:
        """Loads all entries from the store."""
        with open(self.entries_file, "r") as f:
            data = json.load(f)
        return [DBEntry(**v) for v in data.values()]

    def get_entry(self, entry_id: str) -> Optional[DBEntry]:
        """Retrieves an entry by ID."""
        with open(self.entries_file, "r") as f:
            data = json.load(f)
        if entry_id in data:
            return DBEntry(**data[entry_id])
        return None
