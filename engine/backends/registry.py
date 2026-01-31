from typing import List, Dict, Any
from engine.backends.base import IRBackend
from engine.backends.cnf import CNFBackend
from engine.backends.pb import PBBackend
from engine.backends.pb import PBBackend
try:
    from engine.backends.minizinc import MiniZincCoreBackend
    _MINIZINC_AVAILABLE = True
except ImportError:
    _MINIZINC_AVAILABLE = False

class IRBackendRegistry:
    def __init__(self):
        self._backends = {}
        self.register(CNFBackend())
        self.register(PBBackend())
        
        if _MINIZINC_AVAILABLE:
            try:
                self.register(MiniZincCoreBackend())
            except Exception as e:
                print(f"Warning: Failed to initialize MiniZinc Backend: {e}")
        else:
             # Optional: print("MiniZinc backend not available (ImportError)")
             pass

    def register(self, backend: IRBackend):
        self._backends[backend.name] = backend

    def get(self, name: str) -> IRBackend:
        if name not in self._backends:
            raise ValueError(f"Backend '{name}' not found.")
        return self._backends[name]

    def list_backends(self) -> List[str]:
        return list(self._backends.keys())
