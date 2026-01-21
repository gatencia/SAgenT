from typing import List, Dict, Any
from engine.backends.base import IRBackend
from engine.backends.cnf import CNFBackend
from engine.backends.pb import PBBackend
from engine.backends.minizinc import MiniZincCoreBackend

class IRBackendRegistry:
    def __init__(self):
        self._backends = {}
        self.register(CNFBackend())
        self.register(PBBackend())
        self.register(MiniZincCoreBackend())

    def register(self, backend: IRBackend):
        self._backends[backend.name] = backend

    def get(self, name: str) -> IRBackend:
        if name not in self._backends:
            raise ValueError(f"Backend '{name}' not found.")
        return self._backends[name]

    def list_backends(self) -> List[str]:
        return list(self._backends.keys())
