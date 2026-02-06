import os
import time
import fcntl
import json
import hashlib
import shutil
from pathlib import Path
from typing import Any, Optional, Union, Dict
import logging

from Denabase.Denabase.core.errors import DenabaseError

logger = logging.getLogger(__name__)

class LockError(DenabaseError):
    pass

import threading
_local = threading.local()

class FileLock:
    """
    Robust file lock with timeout, exp backoff, and stale lock handling.
    Reentrant within the same thread.
    """
    def __init__(self, lock_file: Union[str, Path], timeout: float = 10.0):
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.fd = None
        
    def acquire(self):
        # Thread-local reentrancy
        if not hasattr(_local, "locks"):
            _local.locks = {}
        
        path_str = str(self.lock_file.absolute())
        if path_str in _local.locks:
            _local.locks[path_str]["count"] += 1
            return

        start_time = time.time()
        delay = 0.01
        
        while True:
            try:
                self.fd = open(self.lock_file, 'w')
                # LOCK_EX: Exclusive, LOCK_NB: Non-blocking
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write PID for stale check
                self.fd.write(str(os.getpid()))
                self.fd.flush()
                
                _local.locks[path_str] = {"count": 1, "fd": self.fd}
                return
            except (IOError, OSError):
                if self.fd:
                    try:
                        self.fd.close()
                    except:
                        pass
                    self.fd = None
                
                # Check for staleness? 
                # If we assume robust OS, flock cleans up on crash. 
                # But sometimes file remains? flock is tied to FD. 
                # If process dies, FD closes, lock releases.
                # So mostly we just wait.
                
                if time.time() - start_time > self.timeout:
                    raise LockError(f"Timeout waiting for lock: {self.lock_file}")
                
                # Exp backoff with jitter?
                time.sleep(delay)
                delay = min(delay * 1.5, 0.5)

    def release(self):
        path_str = str(self.lock_file.absolute())
        if not hasattr(_local, "locks") or path_str not in _local.locks:
            # This thread does not hold the lock, or it was already released
            return

        _local.locks[path_str]["count"] -= 1
        
        if _local.locks[path_str]["count"] <= 0:
            lock_info = _local.locks.pop(path_str)
            fd = lock_info["fd"]
            if fd:
                try:
                    # Truncate/unlink? 
                    # Unlinking lock file while others wait on it is race condition prone in some setups.
                    # Safer to just unlock and close.
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    fd.close()
                except Exception as e:
                    logger.warning(f"Error releasing lock: {e}")
                finally:
                    # Ensure the instance's fd is also cleared if this was the last release
                    if self.fd == fd:
                        self.fd = None
                
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class DiskCache:
    """
    Persistent cache with TTL.
    """
    def __init__(self, cache_dir: Union[str, Path], ttl_seconds: float = 3600):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, key: str) -> Path:
        # Hash key to avoid filename issues
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"
        
    def get(self, key: str) -> Optional[Any]:
        p = self._get_path(key)
        if not p.exists():
            return None
            
        try:
            # Check TTL
            mtime = p.stat().st_mtime
            if time.time() - mtime > self.ttl:
                # Expired
                # We could delete lazily
                return None
                
            with open(p, 'r') as f:
                data = json.load(f)
                return data
        except Exception:
            # Corrupt?
            return None
            
    def set(self, key: str, value: Any):
        p = self._get_path(key)
        # Atomic write
        tmp = p.with_suffix(".tmp")
        try:
            with open(tmp, 'w') as f:
                json.dump(value, f)
            os.replace(tmp, p)
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")
            if tmp.exists():
                try:
                    os.remove(tmp)
                except:
                    pass

    def clear(self):
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir()
