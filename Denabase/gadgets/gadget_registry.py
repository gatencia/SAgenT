from typing import Dict, Type, List, Any
from pathlib import Path
from Denabase.gadgets.gadget_spec import GadgetSpec, ExactlyOneGadget, AtMostOneGadget, KColoringGadget
from Denabase.verify.verifier import CnfVerifier
from Denabase.ir import compile_ir
from Denabase.core.errors import VerificationError
from Denabase.cnf.cnf_types import CnfDocument

class LearnedGadget(GadgetSpec):
    """
    A gadget induced from the database, backed by a verified EntryID.
    """
    name: str = "LearnedGadget"
    desc: str = "Induced from verified motif."
    
    # Instance attributes, not class
    def __init__(self, name: str, entry_id: str, params: Dict[str, Any], family: str = "learned"):
        super().__init__()
        self.name = name # Override
        self.entry_id = entry_id
        self._learned_params = params
        self.family = family

    def build_ir(self, params: Dict[str, Any] = None) -> Any:
        # Learned gadgets are fixed instances for now, or parameterized templates?
        # Requirement says "parameterized by n".
        # If we induce "ExactlyOne" from a specific clique of size 5, do we generalize?
        # Induction logic (Stitch-like) implies finding the pattern.
        # But if we just store the *gadget* result, it might be a specific instance.
        # However, to be a "Gadget", it should be reusable.
        # For this implementation, let's assume LearnedGadget holds the metadata
        # to RECONSTRUCT the gadget logic, or points to a Python generator if we codegen'd it?
        # Or maybe it points to a database entry that IS the gadget definition (like a sub-circuit).
        # "convert to IR gadgets parameterized by n" implies code generation or template usage.
        # For simplicity in this step: LearnedGadget will mimic a standard gadget if the miner identified it as such.
        # If it's a new unknown motif, it's just a fixed verified block.
        # Let's support both: if inferred_type is standard, we delegate. 
        # If not, we just return the IR of the entry?
        # Wait, if we induce "at-most-one", we should just register standard AtMostOneGadget?
        # No, "induce ... convert to IR gadgets".
        # If we find a clique, we recognize it's AMO. 
        # The goal is likely to *confirm* that the DB contains these patterns.
        # Let's implement LearnedGadget as a wrapper that holds the provenance (source entry ID).
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entry_id": self.entry_id,
            "params": self._learned_params,
            "family": self.family
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearnedGadget':
        return cls(
            name=data["name"],
            entry_id=data["entry_id"],
            params=data["params"],
            family=data.get("family", "learned")
        )

class GadgetRegistry:
    """Registry for managing available constraint gadgets."""
    
    def __init__(self):
        self._registry: Dict[str, Type[GadgetSpec]] = {}
        # Store for learned instances (which are objects, not classes)
        self._learned_registry: Dict[str, LearnedGadget] = {}
        
        # Register built-ins
        self.register(ExactlyOneGadget)
        self.register(AtMostOneGadget)
        self.register(KColoringGadget)

    def register(self, gadget_cls: Type[GadgetSpec]):
        """Registers a new gadget class."""
        name = gadget_cls.name
        self._registry[name] = gadget_cls

    def register_learned(self, gadget: LearnedGadget):
        """Registers a learned gadget instance."""
        self._learned_registry[gadget.name] = gadget

    def get(self, name: str) -> GadgetSpec:
        """Returns an instance of the requested gadget."""
        if name in self._registry:
            return self._registry[name]()
        if name in self._learned_registry:
            return self._learned_registry[name]
        raise ValueError(f"Gadget {name} not found.")

    def list_gadgets(self) -> List[str]:
        return list(self._registry.keys()) + list(self._learned_registry.keys())
    
    def save_learned(self, root_dir: Path):
        """Saves all learned gadgets to JSON files in root_dir."""
        # Ensure dir exists? Caller responsibility usually but let's be safe
        # root_dir / "gadgets" usually
        if not root_dir.exists():
            root_dir.mkdir(parents=True, exist_ok=True)
            
        for name, g in self._learned_registry.items():
            path = root_dir / f"{name}.json"
            # Atomic write manually or rely on overwrite
            import json
            with open(path, "w") as f:
                json.dump(g.to_dict(), f, indent=2)

    def load_from_dir(self, root_dir: Path):
        """Loads learned gadgets from directory."""
        import json
        if not root_dir.exists():
            return
            
        for p in root_dir.glob("*.json"):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                g = LearnedGadget.from_dict(data)
                self.register_learned(g)
            except Exception as e:
                # Log?
                print(f"Failed to load gadget from {p}: {e}")

    def run_self_tests(self, verifier: CnfVerifier) -> Dict[str, List[bool]]:
        """
        Runs unit tests for all registered gadgets.
        Returns detailed report: {gadget_name: [pass_test1, pass_test2, ...]}
        """
        results = {}
        
        # Test class-based
        for name, cls in self._registry.items():
            gadget = cls()
            gadget_results = []
            self._run_single_gadget_tests(name, gadget, verifier, gadget_results)
            results[name] = gadget_results
            
        # Test learned based (if they have tests attached?)
        # LearnedGadgets probably don't have unit_tests attribute unless we generate them.
        # For now skip or assume empty list.
        
        return results

    def _run_single_gadget_tests(self, name: str, gadget: GadgetSpec, verifier: CnfVerifier, results: List[bool]):
        for test in gadget.unit_tests:
            try:
                # Build IR
                ir = gadget.build_ir(test.params)
                
                # Compile to CNF
                clauses, varmap = compile_ir(ir)
                max_var = len(varmap)
                if clauses:
                    max_clause_var = max(abs(lit) for clause in clauses for lit in clause)
                    max_var = max(max_var, max_clause_var)
                    
                doc = CnfDocument(num_vars=max_var, clauses=clauses)
                
                # Verify
                res = verifier.verify(doc)
                
                # Check against expectation
                passed = False
                if res.outcome == "PASSED":
                    actual_sat = res.is_satisfiable
                    if actual_sat == test.expected_sat:
                        passed = True
                        
                results.append(passed)
                
            except Exception as e:
                print(f"Gadget {name} test failed validation: {e}")
                results.append(False)

# Global registry instance
registry = GadgetRegistry()
