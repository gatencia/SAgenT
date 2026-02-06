import shutil
import tempfile
import pytest
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir import Exactly, AtMost, VarRef, compile_ir
from Denabase.Denabase.trace import EncodingTrace
from Denabase.Denabase.sleep.sleep_runner import SleepRunner
from Denabase.Denabase.gadgets.gadget_pack import load_pack
from Denabase.Denabase.gadgets.gadget_registry import registry, GadgetRegistry

def test_gadget_pack_sleep_cycle():
    tmp_dir = tempfile.mkdtemp()
    try:
        db = DenaBase.create(tmp_dir)
        
        # 1. Populate DB with Traces containing a motif
        # Motif: Exactly(1, 2 vars) -> AtMost(1, 2 vars)
        for i in range(5):
            c1 = Exactly(k=1, vars=[VarRef(name=f"v{i}_a"), VarRef(name=f"v{i}_b")])
            c2 = AtMost(k=1, vars=[VarRef(name=f"v{i}_c"), VarRef(name=f"v{i}_d")])
            
            trace = EncodingTrace(summary={"goal": f"ex_{i}"})
            
            # Compile with trace
            compile_ir([c1, c2], trace=trace)
            
            # Add to DB
            eid = db.add_ir([c1, c2], family="gen", problem_id=f"p{i}", verify=False)
            db.attach_trace(eid, trace)
            
        # 2. Run Sleep Runner
        print(">> Running Sleep Cycle...")
        runner = SleepRunner(db)
        stats = runner.run_sleep_cycle(min_freq=2, top_k=1, seed=42)
        
        assert stats["candidates_found"] >= 1
        assert stats["verified_count"] >= 1
        version = stats["version"]
        
        print(f">> Version: {version}")
        
        # 3. Check Pack Exists
        pack_dir = Path(tmp_dir) / "gadgets" / "packs" / version
        assert pack_dir.exists()
        assert (pack_dir / "manifest.json").exists()
        assert (pack_dir / "reports").exists()
        
        # 4. Check Registry Loading using Load Pack
        # Clean registry first?
        # We can't easily clear the global registry without affecting other tests,
        # but let's assume we use a fresh registry instance for verification.
        
        fresh_registry = GadgetRegistry()
        # Should be empty of new macros initially (only built-ins)
        initial_count = len(fresh_registry.list_gadgets())
        
        load_pack(pack_dir, fresh_registry)
        
        new_count = len(fresh_registry.list_gadgets())
        print(f">> Registry count: {initial_count} -> {new_count}")
        assert new_count > initial_count
        
        # Verify loaded macro
        macros = [n for n in fresh_registry.list_gadgets() if "macro" in n]
        assert len(macros) > 0
        m_name = macros[0]
        gadget = fresh_registry.get(m_name)
        
        # Verify it works
        params = {
            "step_0_vars": ["x1", "x2"],
            "step_1_vars": ["y1", "y2"]
        }
        ir = gadget.build_ir(params)
        assert len(ir) == 2
        
    finally:
        shutil.rmtree(tmp_dir)
