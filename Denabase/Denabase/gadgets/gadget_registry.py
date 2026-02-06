from typing import Dict, Type, List, Any, Union
from pydantic import Field
from pathlib import Path
from Denabase.Denabase.gadgets.gadget_spec import GadgetSpec, ExactlyOneGadget, AtMostOneGadget, KColoringGadget
from Denabase.Denabase.gadgets.macro_gadget import MacroGadget
from Denabase.Denabase.verify.verifier import CnfVerifier
from Denabase.Denabase.ir import compile_ir
from Denabase.Denabase.core.errors import VerificationError
from Denabase.Denabase.cnf.cnf_types import CnfDocument

class LearnedGadget(GadgetSpec):
    """
    A gadget induced from the database, backed by a verified EntryID.
    """
    name: str = "LearnedGadget"
    desc: str = "Induced from verified motif."
    entry_id: str
    params: Dict[str, Any] = Field(default_factory=dict, alias="_learned_params")
    
    # Instance attributes, not class
    def __init__(self, name: str, entry_id: str, params: Dict[str, Any], family: str = "learned"):
        super().__init__(name=name, entry_id=entry_id, family=family, description="Induced from verified motif.")
        # self.name, self.entry_id automatically set by super if they match field names?
        # But we also have self._learned_params override.
        self._learned_params = params
        self.family = family

    def build_ir(self, params: Dict[str, Any] = None) -> Any:
        """
        Builds IR by delegating to known types or future fallback.
        """
        inferred = self._learned_params.get("inferred_type")
        
        # 1. Delegate to built-ins
        if inferred == "ExactlyOne":
            return ExactlyOneGadget().build_ir(params)
        elif inferred == "AtMostOne":
            return AtMostOneGadget().build_ir(params)
        elif inferred == "KColoring":
            return KColoringGadget().build_ir(params)
            
        # 2. Fallback: Load Fixed CNF from DB
        # Access global registry to get DB handle
        # This is a bit of a coupling hack, but practical for this architecture.
        reg = registry
        if hasattr(reg, "db") and reg.db is not None:
            # We need to map params["vars"] to FixedCNF(vars=...)
            # And we need the clauses from the artifact.
            try:
                # 1. Get artifact
                # We expect entry_id to be stored in self.entry_id
                # The CNF is at entries/EID.cnf or cnf/EID.cnf
                # Using db.get_artifact wrapper if available, or just construct path?
                # Registry has 'db' which is 'DenaBase' instance? Or 'FileStore'?
                # Assuming DenaBase main class.
                # db.store.get_artifact("cnf/{eid}.cnf")
                
                # Check for "vars" param
                if "vars" not in params:
                    raise ValueError(f"LearnedGadget fallback requires 'vars' parameter to map ports.")
                
                # Load CNF content
                # DenaBase has .store
                cnf_content = reg.db.store.get_artifact(f"cnf/{self.entry_id}.cnf")
                if not cnf_content:
                     raise ValueError(f"CNF artifact missing for {self.entry_id}")
                
                # Parse DIMACS
                clauses = []
                if isinstance(cnf_content, bytes):
                    cnf_content = cnf_content.decode("utf-8")
                
                for line in cnf_content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("c") or line.startswith("%") or line.startswith("0") or line.startswith("p"):
                        continue
                    parts = [int(x) for x in line.split() if x != "0"]
                    if parts:
                        clauses.append(parts)
                
                # Build FixedCNF
                from Denabase.Denabase.ir.ir_types import FixedCNF, VarRef
                return FixedCNF(
                    clauses=clauses,
                    vars=[VarRef(name=v) for v in params["vars"]]
                )
                
            except Exception as e:
                raise NotImplementedError(f"LearnedGadget fallback failed for {self.name}: {e}")
        
        raise NotImplementedError(f"LearnedGadget {self.name} of type {inferred} cannot build dynamic IR yet (No DB connection).")

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

# Type alias
RegistryItem = Union[Type[GadgetSpec], LearnedGadget, MacroGadget]

class GadgetRegistry:
    """Registry for managing available constraint gadgets."""
    
    def __init__(self):
        self._registry: Dict[str, Type[GadgetSpec]] = {}
        # Store for learned instances (which are objects, not classes)
        self._learned_registry: Dict[str, LearnedGadget] = {}
        # Store for macro instances
        self._macro_registry: Dict[str, MacroGadget] = {}
        self.db = None # Reference to DenaBase for fallback loading
        
        # Register built-ins
        self.register(ExactlyOneGadget)
        self.register(AtMostOneGadget)
        self.register(KColoringGadget)

    def set_db(self, db):
        """Injects DenaBase instance for lazy loading artifacts."""
        self.db = db

    def register(self, gadget_cls: Type[GadgetSpec]):
        """Registers a new gadget class."""
        # Pydantic models: access default value or instantiate
        try:
             # Fast path: class attribute if set
             name = gadget_cls.model_fields["name"].default
        except:
             name = gadget_cls().name
        
        self._registry[name] = gadget_cls

    def register_learned(self, gadget: LearnedGadget):
        """Registers a learned gadget instance."""
        self._learned_registry[gadget.name] = gadget

    def register_macro(self, gadget: MacroGadget):
        """Registers a macro gadget."""
        self._macro_registry[gadget.name] = gadget

    def get(self, name: str) -> GadgetSpec:
        """Returns an instance of the requested gadget."""
        if name in self._registry:
            return self._registry[name]()
        if name in self._learned_registry:
            return self._learned_registry[name]
        if name in self._macro_registry:
            return self._macro_registry[name]
        raise ValueError(f"Gadget {name} not found.")

    def list_gadgets(self) -> List[str]:
        return list(self._registry.keys()) + list(self._learned_registry.keys()) + list(self._macro_registry.keys())
    
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
                # Log?
                print(f"Failed to load gadget from {p}: {e}")

    def save_macros(self, root_dir: Path):
        """Saves all macro gadgets to JSON files."""
        if not root_dir.exists():
            root_dir.mkdir(parents=True, exist_ok=True)
            
        for name, g in self._macro_registry.items():
            path = root_dir / f"{name}.json"
            import json
            # Serialize model
            with open(path, "w") as f:
                f.write(g.model_dump_json(indent=2))

    def load_macros(self, root_dir: Path):
        """Loads macros from directory."""
        if not root_dir.exists(): return
        
        for p in root_dir.glob("*.json"):
            try:
                import json
                with open(p, "r") as f:
                    data = json.load(f)
                # Ensure it's a macro
                # If we used pydantic serialization, we can load directly
                g = MacroGadget(**data)
                self.register_macro(g)
            except Exception as e:
                print(f"Failed to load macro from {p}: {e}")

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
