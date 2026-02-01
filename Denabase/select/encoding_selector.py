from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from Denabase.profile.profile_types import ConstraintProfile

# Type aliases for strictness
CardinalityEncoding = Literal["pairwise", "sequential", "totalizer_if_available"]
AuxPolicy = Literal["tseitin_min", "tseitin_full"]
VarOrder = Literal["default", "degree_desc"]

class EncodingRecipe(BaseModel):
    """Configuration for translating high-level constraints to CNF."""
    cardinality_encoding: CardinalityEncoding = "sequential"
    aux_policy: AuxPolicy = "tseitin_min"
    var_order: VarOrder = "default"
    symmetry_breaking: bool = False
    notes: List[str] = Field(default_factory=list)

class EncodingSelector:
    """
    Deterministically selects an encoding recipe based on the constraint profile.
    """
    
    def select(self, profile: ConstraintProfile) -> EncodingRecipe:
        recipe = EncodingRecipe()
        
        # 1. Cardinality Encoding Heuristics
        # Check for large cardinality constraints
        max_k = 0
        max_group_size = 0
        
        # In profile.cardinalities, we have lists like "all_k", "all_size"
        # We need to parse them. The profile structure is generic dict.
        # Based on ir_profile implementation: "all_k", "all_size" keys exist.
        
        k_values = profile.cardinalities.get("all_k", [])
        sizes = profile.cardinalities.get("all_size", [])
        
        if k_values:
            max_k = max(k_values)
        if sizes:
            max_group_size = max(sizes)
            
        # Heuristic:
        # If we have large groups or large k, pairwise blows up (O(n^2)).
        # Sequential/Totalizer is O(n*k) or O(n log n).
        
        if max_group_size > 10:
             # For larger groups, avoid pairwise
             recipe.cardinality_encoding = "sequential"
             recipe.notes.append("Selected sequential encoding due to group size > 10.")
        elif max_group_size > 0:
             # Very small groups, pairwise is often efficient and propagation complete
             recipe.cardinality_encoding = "pairwise"
             recipe.notes.append("Selected pairwise encoding for small group sizes.")
        else:
             # No cardinality constraints likely found, defaults fine
             pass
             
        # 2. Variable Ordering & Symmetry
        # Check density from graphish stats
        # profile.graphish might contain "density"
        density = profile.graphish.get("density", 0.0)
        
        if density > 0.5:
             # Very dense problems might benefit from symmetry breaking if graph automorphisms exist
             # This is a weak heuristic but serves as a placeholder.
             recipe.symmetry_breaking = True
             recipe.notes.append("Enabled symmetry breaking due to high graph density.")
             
        return recipe
