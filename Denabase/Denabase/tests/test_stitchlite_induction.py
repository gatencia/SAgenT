import shutil
import tempfile
import pytest
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir import Exactly, AtMost, VarRef, compile_ir
from Denabase.Denabase.trace import EncodingTrace
from Denabase.Denabase.induce import StitchLiteMiner
from Denabase.Denabase.gadgets.gadget_registry import registry
from Denabase.Denabase.gadgets.macro_gadget import MacroGadget

def test_stitchlite_induction():
    # 1. Setup Temp DB
    tmp_dir = tempfile.mkdtemp()
    try:
        db = DenaBase.create(tmp_dir)
        
        # 2. Create Synthetic Corpus with Repeated Motif
        # Motif: Exactly(1, [a,b]) -> AtMost(1, [c,d])
        # We will add this sequence 3 times (freq=3 >= min_freq=2)
        
        for i in range(3):
            trace = EncodingTrace(summary={"goal": f"ex_{i}"})
            
            # Constraints
            v1 = [VarRef(name=f"v{i}_a"), VarRef(name=f"v{i}_b")]
            v2 = [VarRef(name=f"v{i}_c"), VarRef(name=f"v{i}_d")]
            
            c1 = Exactly(k=1, vars=v1)
            c2 = AtMost(k=1, vars=v2)
            
            # Compile with trace
            compile_ir([c1, c2], trace=trace)
            
            # Add to DB
            eid = db.add_ir([c1, c2], family="gen", problem_id=f"p{i}", verify=False)
            db.attach_trace(eid, trace)
            
        # 3. Run Miner
        print(">> Running Miner...")
        miner = StitchLiteMiner(db)
        macros = miner.mine(min_freq=2, top_k=1, max_ngram=2)
        
        assert len(macros) >= 1
        macro = macros[0]
        print(">> Macro found:", macro.name)
        
        assert isinstance(macro, MacroGadget)
        assert len(macro.ir_template) == 2
        # Check inferred types
        types = [n["type"] for n in macro.ir_template]
        assert "Exactly" in types
        assert "AtMost" in types
        
        # 4. Registry Integration
        print(">> Registering macro...")
        registry.register_macro(macro)
        save_path = Path(tmp_dir) / "macros"
        print(">> Saving macros...")
        registry.save_macros(save_path)
        print(">> Macros saved.")
        
        # Clear and reload
        from Denabase.Denabase.gadgets.gadget_registry import GadgetRegistry
        reg2 = GadgetRegistry()
        reg2.load_macros(save_path)
        
        assert macro.name in reg2.list_gadgets()
        loaded = reg2.get(macro.name)
        assert isinstance(loaded, MacroGadget)
        
        # 5. Verify Build IR
        print(">> verifying build_ir...")
        params = {
            "step_0_vars": ["x", "y"], 
            "step_1_vars": ["z", "w"]
        }
        
        ir_out = loaded.build_ir(params)
        assert len(ir_out) == 2
        
        # Compile it to ensure validity
        print(">> compiling result...")
        clauses, varmap = compile_ir(ir_out)
        assert len(clauses) > 0
        print(">> DONE")
        
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_stitchlite_induction()
