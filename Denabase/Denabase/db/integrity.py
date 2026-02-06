from pathlib import Path
from typing import List, Dict, Any, Set
import json
import logging
import hashlib

from Denabase.Denabase.db.store import FileStore
from Denabase.Denabase.cnf.cnf_io import load_cnf
from Denabase.Denabase.profile.profile_types import profile_hash
from Denabase.Denabase.graph.fingerprints import compute_fingerprint, Fingerprint
from Denabase.Denabase.profile.cnf_profile import ConstraintProfile

logger = logging.getLogger(__name__)

class IntegrityReport:
    def __init__(self):
        self.missing_files: List[str] = []
        self.corrupt_json: List[str] = []
        self.hash_mismatches: List[str] = []
        self.fingerprint_mismatches: List[str] = []
        self.valid_entries: List[str] = []

    def is_clean(self) -> bool:
        return not (self.missing_files or self.corrupt_json or self.hash_mismatches or self.fingerprint_mismatches)

    def summary(self) -> str:
        if self.is_clean():
            return f"Integrity check passed. {len(self.valid_entries)} valid entries."
        return (
            f"Integrity check FAILED.\n"
            f"Valid: {len(self.valid_entries)}\n"
            f"Missing Files: {len(self.missing_files)}\n"
            f"Corrupt JSON: {len(self.corrupt_json)}\n"
            f"Hash Mismatches: {len(self.hash_mismatches)}\n"
            f"Fingerprint Mismatches: {len(self.fingerprint_mismatches)}"
        )

class IntegrityChecker:
    def __init__(self, store: FileStore):
        self.store = store

    def scan(self) -> IntegrityReport:
        """Scans the database for consistency issues."""
        report = IntegrityReport()
        entries = self.store.load_entries()
        
        for entry in entries:
            eid = entry.id
            is_valid = True
            
            # 1. Check artifact existence and JSON validity
            for key, rel_path in entry.paths.items():
                abs_path = self.store.root / rel_path
                if not abs_path.exists():
                    report.missing_files.append(f"{eid}: {key} file missing at {rel_path}")
                    is_valid = False
                    continue
                
                # If JSON, verify readability
                if abs_path.suffix == ".json":
                    try:
                        with open(abs_path, "r") as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        report.corrupt_json.append(f"{eid}: {key} file corrupt at {rel_path}")
                        is_valid = False

            if not is_valid:
                continue

            # 2. Deep verification (hashes and re-computation)
            try:
                # Load artifacts
                cnf_path = self.store.root / entry.paths["cnf"]
                prof_path = self.store.root / entry.paths["profile"]
                fp_path = self.store.root / entry.paths["fingerprint"]
                
                # CNF Content Hash
                try:
                    doc = load_cnf(cnf_path)
                except Exception as e:
                    report.corrupt_json.append(f"{eid}: CNF parse error: {e}")
                    is_valid = False
                    continue

                # Verify stored content hash if available
                if "content_hash" in entry.hashes:
                    actual_hash = doc.content_hash()
                    if actual_hash != entry.hashes["content_hash"]:
                        report.hash_mismatches.append(f"{eid}: CNF content_hash mismatch")
                        is_valid = False

                # Profile
                with open(prof_path, "r") as f:
                    prof_data = json.load(f)
                    actual_prof_hash = profile_hash(ConstraintProfile(**prof_data))
                
                # Recompute Fingerprint
                prof_obj = ConstraintProfile(**prof_data)
                recomputed_fp = compute_fingerprint(doc, prof_obj)
                
                with open(fp_path, "r") as f:
                    stored_fp_data = json.load(f)
                    stored_fp = Fingerprint(**stored_fp_data)
                
                if recomputed_fp.content_hash != stored_fp.content_hash:
                     report.fingerprint_mismatches.append(f"{eid}: Computed content_hash {recomputed_fp.content_hash} != stored {stored_fp.content_hash}")
                     is_valid = False
                
                if recomputed_fp.wl_hash != stored_fp.wl_hash:
                     report.fingerprint_mismatches.append(f"{eid}: Computed wl_hash != stored")
                     is_valid = False

            except Exception as e:
                logger.error(f"Error checking entry {eid}: {e}")
                report.corrupt_json.append(f"{eid}: Exception during deep check: {e}")
                is_valid = False

            if is_valid:
                report.valid_entries.append(eid)
                
        return report

    def repair_indexes(self, embedder=None, index=None) -> int:
        """
        Rebuilds secondary indexes (signatures, profiles, vector) from valid entries.
        Returns number of indexed entries.
        """
        report = self.scan()
        signatures = {}
        profiles_inverted = {}
        
        for eid in report.valid_entries:
            try:
                entry = self.store.get_entry_record(eid)
                if not entry: continue
                
                fp_path = self.store.root / entry.paths["fingerprint"]
                with open(fp_path, "r") as f:
                    fp = Fingerprint(**json.load(f))
                
                # Update signatures
                k = fp.signature_key
                if k not in signatures: signatures[k] = []
                signatures[k].append(eid)
                
                # Update inverted profiles
                tokens = [f"sig:{fp.signature_key}", f"wl:{fp.wl_hash}"]
                
                for t in tokens:
                    if t not in profiles_inverted: profiles_inverted[t] = []
                    profiles_inverted[t].append(eid)
                    
                # Update Vector Index if provided
                if index and embedder:
                    cnf_path = self.store.root / entry.paths["cnf"]
                    prof_path = self.store.root / entry.paths["profile"]
                    
                    doc = load_cnf(cnf_path)
                    with open(prof_path, "r") as f:
                         prof = ConstraintProfile(**json.load(f))
                    
                    vec = embedder.embed(doc, prof)
                    index.add([vec.tolist()], [eid])
                    
            except Exception as e:
                logger.error(f"Failed to re-index {eid}: {e}")
                continue
        
        self.store._atomic_write_json(self.store.signatures_file, signatures)
        self.store._atomic_write_json(self.store.profiles_inv_file, profiles_inverted)
        
        return len(report.valid_entries)
