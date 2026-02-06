from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import json
from pathlib import Path
import hashlib

from Denabase.Denabase.gadgets.macro_gadget import MacroGadget
from Denabase.Denabase.gadgets.gadget_registry import GadgetRegistry, registry

class GadgetPackManifest(BaseModel):
    """
    Manifest for a versioned collection of induced gadgets.
    """
    version: str # e.g. YYYYMMDD_HHMMSS
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    macros: List[str] = Field(default_factory=list) # List of macro names included
    learned: List[str] = Field(default_factory=list) # List of learned gadget names included
    stats: Dict[str, Any] = Field(default_factory=dict) # e.g. {num_macros: 5, total_gain: 120}
    git_sha: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Computes a hash of the manifest content."""
        dump = self.model_dump_json(exclude={"created_at"})
        return hashlib.sha256(dump.encode()).hexdigest()

def save_pack(root_dir: Path, registry: GadgetRegistry, verification_reports: Dict[str, Any] = {}) -> str:
    """
    Saves the current registry state as a versioned pack.
    Returns the pack version string.
    """
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pack_dir = root_dir / "gadgets" / "packs" / version
    pack_dir.mkdir(parents=True, exist_ok=True)
    
    # Save macros
    macro_names = []
    for name, g in registry._macro_registry.items():
        fname = f"{name}.json"
        with open(pack_dir / fname, "w") as f:
            f.write(g.model_dump_json(indent=2))
        macro_names.append(name)
        
    # Save learned (optional, if we want to pack them too)
    learned_names = []
    # For now, we only pack macros as they are the reusable library entities.
    # Learned gadgets are more instance-specific.
    
    # Validation reports
    if verification_reports:
        reports_dir = pack_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        for name, report in verification_reports.items():
            if name in macro_names:
                with open(reports_dir / f"{name}_report.json", "w") as f:
                    json.dump(report, f, indent=2)
                    
    # Create Manifest
    manifest = GadgetPackManifest(
        version=version,
        macros=sorted(macro_names),
        learned=sorted(learned_names),
        stats={
            "num_macros": len(macro_names),
            "verification_coverage": len(verification_reports) / len(macro_names) if macro_names else 0
        }
    )
    
    with open(pack_dir / "manifest.json", "w") as f:
        f.write(manifest.model_dump_json(indent=2))
        
    return version

def load_pack(pack_dir: Path, registry: GadgetRegistry = registry):
    """
    Loads a gadget pack into the registry.
    """
    if not pack_dir.exists():
        raise FileNotFoundError(f"Pack directory {pack_dir} not found")
        
    with open(pack_dir / "manifest.json", "r") as f:
        data = json.load(f)
    manifest = GadgetPackManifest(**data)
    
    # Load macros
    for m in manifest.macros:
        p = pack_dir / f"{m}.json"
        if p.exists():
            with open(p, "r") as f:
                data = json.load(f)
            g = MacroGadget(**data)
            registry.register_macro(g)
            
    # Load learned (if any)
    # ...
