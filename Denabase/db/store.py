import json
import os
import time
import fcntl
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
from contextlib import contextmanager

from Denabase.db.schema import EntryRecord, DBMeta, VerificationRecord, EntryMeta
from Denabase.core.errors import DenabaseError
from Denabase.core.cache import FileLock
from Denabase.trace import EncodingTrace, save_trace_json, load_trace_json

BS = 65536

class LockError(DenabaseError):
    pass

class FileStore:
    """
    Robust file-based store for Denabase.
    Handles atomic writes, directory structure, and basic locking.
    """
    
    def __init__(self, db_root: Path):
        self.root = db_root
        self.lock_file = self.root / "lock"
        
        # Directory structure
        self.dirs = {
            "entries": self.root / "entries",
            "cnf": self.root / "cnf",
            "ir": self.root / "ir",
            "profiles": self.root / "profiles",
            "fingerprints": self.root / "fingerprints",
            "verification": self.root / "verification",
            "provenance": self.root / "provenance",
            "telemetry": self.root / "telemetry",
            "indexes": self.root / "indexes",
            "traces": self.root / "traces"
        }
        
        # Create dirs
        for p in self.dirs.values():
            p.mkdir(parents=True, exist_ok=True)
            
        # Initialize meta if fresh
        self.meta_file = self.root / "meta.json"
        if not self.meta_file.exists():
            self._atomic_write_json(self.meta_file, DBMeta().model_dump())
            
        # Initialize indexes if fresh
        self.signatures_file = self.dirs["indexes"] / "signatures.json"
        if not self.signatures_file.exists():
            self._atomic_write_json(self.signatures_file, {})
            
        self.profiles_inv_file = self.dirs["indexes"] / "profiles_inverted.json"
        if not self.profiles_inv_file.exists():
            self._atomic_write_json(self.profiles_inv_file, {})

        # Bulk state
        self._in_bulk = False
        self._bulk_meta = None
        self._bulk_sigs = None
        self._bulk_profs = None

    def begin_bulk(self):
        """Enter bulk mode: load indexes into memory and defer writes."""
        if self._in_bulk: return
        self._in_bulk = True
        self._bulk_meta = DBMeta(**self._read_json(self.meta_file))
        self._bulk_sigs = self._read_json(self.signatures_file)
        self._bulk_profs = self._read_json(self.profiles_inv_file)

    def end_bulk(self):
        """Exit bulk mode: flush all buffers to disk."""
        if not self._in_bulk: return
        
        with self._lock():
            self._atomic_write_json(self.meta_file, self._bulk_meta.model_dump())
            self._atomic_write_json(self.signatures_file, self._bulk_sigs)
            self._atomic_write_json(self.profiles_inv_file, self._bulk_profs)
            
        self._in_bulk = False
        self._bulk_meta = None
        self._bulk_sigs = None
        self._bulk_profs = None

    def get_meta(self) -> DBMeta:
        """Returns metadata, from buffer if in bulk mode."""
        if self._in_bulk:
            return self._bulk_meta
        return DBMeta(**self._read_json(self.meta_file))

    def set_meta(self, meta: DBMeta):
        """Saves metadata, to buffer if in bulk mode."""
        if self._in_bulk:
            self._bulk_meta = meta
        else:
            with self._lock():
                self._atomic_write_json(self.meta_file, meta.model_dump())

    @contextmanager
    def _lock(self, timeout: float = 10.0):
        """Robust file locking."""
        lock = FileLock(self.lock_file, timeout=timeout)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _atomic_write_json(self, path: Path, data: Any):
        """Writes JSON atomically using a temp file."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        
    def _read_json(self, path: Path) -> Any:
        with open(path, "r") as f:
            return json.load(f)

    def put_entry(self, 
                  record: EntryRecord,
                  artifacts: Dict[str, Any],
                  verification_record: Optional[VerificationRecord] = None) -> None:
        """
        Saves an entry and its artifacts. Artifacts dict maps 'cnf', 'ir', etc. to their data object.
        Data objects can be Pydantic models (use model_dump) or raw strings/bytes.
        If verification_record is provided, it is added/overwrites artifact['verification'].
        """
        with self._lock():
            eid = record.id
            
            # Merge verification record if provided
            if verification_record:
                artifacts["verification"] = verification_record
            
            # 1. Write artifacts
            # Mappings from artifact key to (subdir_name, file_ext)
            mapping = {
                "cnf": ("cnf", ".cnf"),
                "ir": ("ir", ".json"),
                "profile": ("profiles", ".json"),
                "fingerprint": ("fingerprints", ".json"),
                "verification": ("verification", ".json"),
                "provenance": ("provenance", ".json"),
                "telemetry": ("telemetry", ".json"),
                "trace": ("traces", ".json")
            }
            
            paths_update = {}
            hashes_update = {}
            
            for key, (subdir, ext) in mapping.items():
                if key in artifacts and artifacts[key] is not None:
                    data = artifacts[key]
                    fname = f"{eid}{ext}"
                    fpath = self.dirs[subdir] / fname
                    
                    # Serialize
                    if hasattr(data, "model_dump"):
                        content = json.dumps(data.model_dump(), indent=2)
                        mode = "w"
                    elif isinstance(data, (dict, list)):
                        # Helper to serialize Pydantic models inside lists/dicts
                        def default_serializer(obj):
                            if hasattr(obj, "model_dump"):
                                return obj.model_dump()
                            # Handle set conversion too while we're at it
                            if isinstance(obj, set):
                                return list(obj)
                            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

                        content = json.dumps(data, indent=2, default=default_serializer)
                        mode = "w"
                    elif isinstance(data, str):
                        content = data
                        mode = "w"
                    elif isinstance(data, bytes):
                        content = data
                        mode = "wb"
                    else:
                        continue # Skip unknown?
                        
                    # Write
                    tmp = fpath.with_suffix(".tmp")
                    with open(tmp, mode) as f:
                        f.write(content)
                    os.replace(tmp, fpath)
                    
                    paths_update[key] = f"{subdir}/{fname}"
                    # Could calculate hash here if needed
                    
            record.paths.update(paths_update)
            
            # 2. Write main entry record
            self._atomic_write_json(self.dirs["entries"] / f"{eid}.json", record.model_dump())
            
            # 3. Update global counts
            meta = self.get_meta()
            meta.counts["entries"] = meta.counts.get("entries", 0) + 1
            self.set_meta(meta)

    def save_trace(self, entry_id: str, trace: EncodingTrace) -> str:
        """Saves a trace and returns relative path."""
        with self._lock():
            fname = f"{entry_id}.json"
            fpath = self.dirs["traces"] / fname
            
            # Use trace module helper
            save_trace_json(trace, fpath)
            
            return f"traces/{fname}"

    def load_trace(self, entry_id: str) -> Optional[EncodingTrace]:
        """Loads a trace for an entry."""
        fpath = self.dirs["traces"] / f"{entry_id}.json"
        if not fpath.exists():
            return None
        return load_trace_json(fpath)

    def update_indexes(self, entry_id: str, signature_key: str, profile_tokens: List[str]):
        """Updates the inverted indexes with a new entry."""
        if self._in_bulk:
            # Update Signatures Buffer
            if signature_key not in self._bulk_sigs:
                self._bulk_sigs[signature_key] = []
            if entry_id not in self._bulk_sigs[signature_key]:
                self._bulk_sigs[signature_key].append(entry_id)
            
            # Update Profile Tokens Buffer
            for token in profile_tokens:
                if token not in self._bulk_profs:
                    self._bulk_profs[token] = []
                if entry_id not in self._bulk_profs[token]:
                    self._bulk_profs[token].append(entry_id)
            return

        with self._lock():
            # Signatures
            sigs = self._read_json(self.signatures_file)
            if signature_key not in sigs:
                sigs[signature_key] = []
            if entry_id not in sigs[signature_key]:
                sigs[signature_key].append(entry_id)
            self._atomic_write_json(self.signatures_file, sigs)
            
            # Profile Tokens
            profs = self._read_json(self.profiles_inv_file)
            for token in profile_tokens:
                if token not in profs:
                    profs[token] = []
                if entry_id not in profs[token]:
                    profs[token].append(entry_id)
            self._atomic_write_json(self.profiles_inv_file, profs)

    def get_entry_record(self, entry_id: str) -> Optional[EntryRecord]:
        p = self.dirs["entries"] / f"{entry_id}.json"
        if not p.exists():
            return None
        return EntryRecord(**self._read_json(p))
    
    def get_verification_record(self, entry_id: str) -> Optional[VerificationRecord]:
        """Retrieves the verification record for an entry, if available."""
        p = self.dirs["verification"] / f"{entry_id}.json"
        if not p.exists():
            return None
        try:
            return VerificationRecord(**self._read_json(p))
        except Exception:
            return None

    def load_entries(self) -> List[EntryRecord]:
        """Loads all entry records from the entries directory."""
        records = []
        # Filter for .json files
        for p in self.dirs["entries"].glob("*.json"):
            try:
                records.append(EntryRecord(**self._read_json(p)))
            except Exception as e:
                # Log or skip? For integrity check, maybe we want to know?
                # But here we just return loadable ones?
                # Integrity Checker will likely scan by listing files if explicit Scan method did that.
                # But IntegrityChecker current scan() calls load_entries().
                # If load fails, scan might miss it?
                # Integrity checker should probably iterate files itself to detect corrupt entry records.
                # But for now, let's just return what we can load.
                continue
        return records

    def get_artifact(self, path_rel: str) -> Any:
        p = self.root / path_rel
        if not p.exists():
            return None
        # Naive: try json load, fallback read text
        try:
            return self._read_json(p)
        except:
             try:
                 return p.read_text()
             except:
                 return p.read_bytes()

    def get_candidate_ids_by_signature(self, signature_key: str) -> List[str]:
        sigs = self._read_json(self.signatures_file)
        return sigs.get(signature_key, [])

    def get_candidate_ids_by_profile_tokens(self, tokens: List[str]) -> Set[str]:
        profs = self._read_json(self.profiles_inv_file)
        candidates = set()
        for t in tokens:
            candidates.update(profs.get(t, []))
        return candidates

    def iter_entry_metas(self):
        """Yields EntryMeta objects (memory efficient)."""
        for p in self.dirs["entries"].glob("*.json"):
            try:
                data = self._read_json(p)
                yield EntryMeta(**data["meta"])
            except Exception:
                continue

    def load_all_profiles(self) -> Dict[str, Any]:
        """Loads all profiles for alpha training. Returns dict {eid: profile_dict}."""
        # Profiles stored in artifacts. This might be slow if many files.
        # But we need them for features.
        # Ideally we'd have a summary index, but we don't yet.
        profiles = {}
        for p in self.dirs["profiles"].glob("*.json"):
            # Filename is {eid}.json
            eid = p.stem
            try:
                profiles[eid] = self._read_json(p)
            except:
                continue
        return profiles
