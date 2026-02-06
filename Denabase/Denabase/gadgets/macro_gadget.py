from typing import List, Dict, Any, Union
from Denabase.Denabase.gadgets.gadget_spec import GadgetSpec
from Denabase.Denabase.ir.ir_types import (
    BoolExpr, Cardinality, VarRef, Lit, Not, And, Or, Exactly, AtMost, AtLeast, Imp, Iff, Xor
)
from pydantic import BaseModel
import copy

class MacroGadget(GadgetSpec):
    """
    A template-based gadget composed of multiple atomic constraints.
    Can be parameterized.
    """
    name: str = "Macro"
    family: str = "macro"
    description: str = "A reusable macro induced from execution traces."
    
    # Template: List of dicts representing the IR nodes structure
    # With placeholders like "$n", "$vars"
    ir_template: List[Dict[str, Any]]
    
    def build_ir(self, params: Dict[str, Any]) -> List[Union[BoolExpr, Cardinality]]:
        self.validate_params(params)
        constraints = []
        
        for template_node in self.ir_template:
            # Deep copy to safe substitution
            node_def = copy.deepcopy(template_node)
            ir_obj = self._instantiate_node(node_def, params)
            if ir_obj:
                constraints.append(ir_obj)
                
        return constraints

    def _instantiate_node(self, node_def: Dict[str, Any], params: Dict[str, Any]) -> Any:
        # Resolve parameters in the dict first
        resolved_def = self._resolve_params(node_def, params)
        node_type = resolved_def.get("type")
        
        if not node_type: return None
        
        # Instantiate based on type
        # Cardinality
        if node_type == "Exactly":
            return Exactly(k=resolved_def.get("k", 1), vars=self._parse_vars(resolved_def.get("vars", [])))
        elif node_type == "AtMost":
            return AtMost(k=resolved_def.get("k", 1), vars=self._parse_vars(resolved_def.get("vars", [])))
        elif node_type == "AtLeast":
            return AtLeast(k=resolved_def.get("k", 1), vars=self._parse_vars(resolved_def.get("vars", [])))
            
        # Boolean
        elif node_type == "And":
            # Recurse if terms provided in template?
            # StitchLite v1 might assume flat or simple structural
            # For now assume flat list of constraints in template, so top level is usually cardinality or basic logic
            return None # To implement if needed
            
        return None

    def _resolve_params(self, obj: Any, params: Dict[str, Any]) -> Any:
        if isinstance(obj, dict):
            return {k: self._resolve_params(v, params) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_params(v, params) for v in obj]
        elif isinstance(obj, str):
            if obj.startswith("$") and obj[1:] in params:
                return params[obj[1:]]
        return obj

    def _parse_vars(self, var_list: List[str]) -> List[VarRef]:
        # Handle var generation from N?
        # If we have parameter n=5, we might need to generate v_0...v_4?
        # For v1 StitchLite, let's assume vars are passed explicitly unless N logic is added.
        # Requirement says "Parameterize only when safe (e.g., repeated group sizes can be placeholder N)".
        # For variables, usually caller passes a list of vars.
        return [VarRef(name=v) for v in var_list]
