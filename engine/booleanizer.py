from typing import Dict, Any, List, Optional
from engine.vars import VarManager

class Booleanizer:
    """
    Manages the translation between FlatZinc variables and SAT literals.
    Handles strict Boolean variables and Integer variables using Order Encoding.
    Delegates ID allocation to VarManager.
    """
    def __init__(self, var_manager: Optional[VarManager] = None):
        self.var_manager = var_manager if var_manager else VarManager()
        self.var_map: Dict[str, Any] = {}
        # Reverse map for decoding (Literal -> Info) could be useful later
        self.literal_to_info: Dict[int, Any] = {}

    def _get_new_literal(self) -> int:
        return self.var_manager.fresh(prefix="mzn", namespace="fzn")

    def register_bool(self, name: str) -> int:
        """
        Registers a boolean variable.
        Returns the assigned SAT literal.
        """
        if name in self.var_map:
            raise ValueError(f"Variable {name} already registered.")
        
        # Use declared variable ID from VarManager if allowed, or fresh?
        # MiniZinc variables map 1:1 to SAT vars typically.
        # We use declare() to ensure consistent naming if name allows.
        lit = self.var_manager.declare(name)
        
        info = {
            "type": "bool",
            "literal": lit,
            "name": name
        }
        self.var_map[name] = info
        self.literal_to_info[lit] = info
        return lit

    def register_int(self, name: str, min_val: int, max_val: int) -> Dict[int, int]:
        """
        Registers an integer variable with domain [min_val, max_val] using Order Encoding.
        Returns a dictionary mapping value k -> literal_id (representing x <= k).
        
        Note:
        - We generate literals for k in [min_val, max_val - 1].
        - (x <= max_val) is syntactically True (no literal needed usually, or handled as const).
        - (x <= min_val - 1) is syntactically False.
        """
        if name in self.var_map:
            raise ValueError(f"Variable {name} already registered.")
        
        if min_val > max_val:
            raise ValueError(f"Invalid domain [{min_val}, {max_val}] for {name}")

        literals = {}
        # Generate literals for x <= k
        for k in range(min_val, max_val):
            lit = self._get_new_literal()
            literals[k] = lit
            
            # Metadata for decoding identifying this specific rank literal
            self.literal_to_info[lit] = {
                "type": "int_rank",
                "name": name,
                "value": k,  # means "name <= k"
                "domain": (min_val, max_val)
            }

        info = {
            "type": "int",
            "domain": (min_val, max_val),
            "rank_literals": literals, # Map k -> lit
            "name": name
        }
        self.var_map[name] = info
        return literals

    def get_bool_literal(self, name: str) -> int:
        """Retrieve literal for a boolean variable."""
        info = self.var_map.get(name)
        if not info or info["type"] != "bool":
            raise ValueError(f"Unknown or non-boolean variable: {name}")
        return info["literal"]

    def get_int_literals(self, name: str) -> Dict[int, int]:
        """Retrieve rank literals for an integer variable."""
        info = self.var_map.get(name)
        if not info or info["type"] != "int":
            raise ValueError(f"Unknown or non-integer variable: {name}")
        return info["rank_literals"]
