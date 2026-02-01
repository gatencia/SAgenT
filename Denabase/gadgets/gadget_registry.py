from typing import Dict, Type, List, Any
from Denabase.gadgets.gadget_spec import GadgetSpec, ExactlyOneGadget, AtMostOneGadget, KColoringGadget
from Denabase.verify.verifier import CnfVerifier
from Denabase.ir import compile_ir
from Denabase.core.errors import VerificationError
from Denabase.cnf.cnf_types import CnfDocument

class GadgetRegistry:
    """Registry for managing available constraint gadgets."""
    
    def __init__(self):
        self._registry: Dict[str, Type[GadgetSpec]] = {}
        # Register built-ins
        self.register(ExactlyOneGadget)
        self.register(AtMostOneGadget)
        self.register(KColoringGadget)

    def register(self, gadget_cls: Type[GadgetSpec]):
        """Registers a new gadget class."""
        # Instantiate to get name if it's not a class property, but here it is class prop.
        # But wait, GadgetSpec defines name as annotation. It should be assigned in subclass.
        # Using instancing to be safe? Or just access class attr.
        # In python class attrs are accessible directly.
        name = gadget_cls.name
        self._registry[name] = gadget_cls

    def get(self, name: str) -> GadgetSpec:
        """Returns an instance of the requested gadget."""
        if name not in self._registry:
            raise ValueError(f"Gadget {name} not found.")
        return self._registry[name]()

    def list_gadgets(self) -> List[str]:
        return list(self._registry.keys())

    def run_self_tests(self, verifier: CnfVerifier) -> Dict[str, List[bool]]:
        """
        Runs unit tests for all registered gadgets.
        Returns detailed report: {gadget_name: [pass_test1, pass_test2, ...]}
        """
        results = {}
        
        for name, cls in self._registry.items():
            gadget = cls()
            gadget_results = []
            
            for test in gadget.unit_tests:
                try:
                    # Build IR
                    ir = gadget.build_ir(test.params)
                    
                    # Compile to CNF
                    clauses, varmap = compile_ir(ir)
                    # varmap only has base vars, but compilation creates aux vars.
                    # We need to find the max variable used.
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
                            
                    gadget_results.append(passed)
                    
                except Exception as e:
                    print(f"Gadget {name} test failed validation: {e}")
                    gadget_results.append(False)
                    
            results[name] = gadget_results
            
        return results

# Global registry instance
registry = GadgetRegistry()
