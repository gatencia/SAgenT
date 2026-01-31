from typing import Dict, List, Optional, Set

class VarManager:
    """
    Centralized manager for SAT variable allocation.
    Ensures deterministic ID assignment and prevents collisions between
    user-defined variables and backend-generated auxiliary variables.
    """
    def __init__(self):
        self._var_map: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        self._next_id: int = 1
        self._aux_vars: Set[int] = set()
    
    @property
    def next_var_id(self) -> int:
        return self._next_id
        
    @property
    def max_id(self) -> int:
        return self._next_id - 1

    def declare(self, name: str) -> int:
        """
        Declare a user variable. Returns existing ID if already declared.
        """
        if name in self._var_map:
            return self._var_map[name]
        
        vid = self._next_id
        self._var_map[name] = vid
        self._id_to_name[vid] = name
        self._next_id += 1
        return vid

    def fresh(self, prefix: str = "aux", namespace: str = "default") -> int:
        """
        Allocate a fresh auxiliary variable.
        """
        name = f"::{namespace}::{prefix}_{self._next_id}"
        vid = self._next_id
        self._var_map[name] = vid
        self._id_to_name[vid] = name
        self._aux_vars.add(vid)
        self._next_id += 1
        return vid

    def reserve_block(self, k: int, prefix: str = "aux", namespace: str = "default") -> List[int]:
        """
        Reserve a block of k fresh variables.
        """
        vids = []
        for _ in range(k):
            vids.append(self.fresh(prefix, namespace))
        return vids

    def get_var_map(self) -> Dict[str, int]:
        return self._var_map.copy()

    def get_id_to_name(self) -> Dict[int, str]:
        return self._id_to_name.copy()
    
    def get_aux_vars(self) -> Set[int]:
        return self._aux_vars.copy()
