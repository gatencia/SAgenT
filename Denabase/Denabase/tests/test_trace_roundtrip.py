import shutil
import tempfile
from pathlib import Path
import pytest
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir.ir_types import Exactly, VarRef
from Denabase.Denabase.ir.ir_compile import compile_ir
from Denabase.Denabase.trace import EncodingTrace, TraceEvent

def test_trace_roundtrip():
    # 1. Setup Temp DB
    tmp_dir = tempfile.mkdtemp()
    try:
        db = DenaBase.create(tmp_dir)
        
        # 2. Compile IR to generate trace
        # IR: Exactly(1, [A, B, C])
        vars = [VarRef(name=f"v{i}") for i in range(3)]
        expr = Exactly(k=1, vars=vars)
        
        trace = EncodingTrace(summary={"goal": "test_trace"})
        compile_ir(expr, trace=trace)
        
        # Verify trace content pre-save
        assert len(trace.events) >= 2 # IR_NODE + CNF_EMIT
        assert trace.events[0].kind == "IR_NODE"
        assert trace.events[0].payload["type"] == "Exactly"
        assert trace.events[-1].kind == "CNF_EMIT"
        assert trace.events[-1].payload["clauses"] > 0
        
        # 3. Add Dummy Entry
        # We need an entry ID to attach trace to.
        # Use add_verified_dimacs? No, add_ir is easier.
        eid = db.add_ir(expr, family="test", problem_id="p1", verify=False)
        
        # 4. Attach Trace
        db.attach_trace(eid, trace)
        
        # 5. Verify Persistence
        # Re-open DB
        db2 = DenaBase.open(tmp_dir)
        loaded_trace = db2.get_trace(eid)
        
        assert loaded_trace is not None
        assert loaded_trace.summary["goal"] == "test_trace"
        assert len(loaded_trace.events) == len(trace.events)
        assert loaded_trace.events[0].kind == "IR_NODE"
        
        # Verify Meta
        rec = db2.store.get_entry_record(eid)
        assert rec.meta.has_trace is True
        assert rec.meta.trace_path == f"traces/{eid}.json"
        
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_trace_roundtrip()
