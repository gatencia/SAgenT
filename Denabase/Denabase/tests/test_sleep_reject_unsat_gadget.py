
import pytest
import shutil
import tempfile
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.sleep.sleep_runner import SleepRunner
from Denabase.Denabase.gadgets.macro_gadget import MacroGadget

def test_reject_unsat_gadget():
    # 1. Setup
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "db"
    
    try:
        db = DenaBase.create(db_path)
        runner = SleepRunner(db)
        
        # 2. Create UNSAT Macro
        # AtMost(0, x) means x is False
        # AtLeast(1, x) means x is True
        # Together -> False
        unsat_template = [
            {"type": "AtMost", "k": 0, "vars": "$vars"},
            {"type": "AtLeast", "k": 1, "vars": "$vars"}
        ]
        
        gadget = MacroGadget(
            name="Contradiction",
            description="Always False",
            ir_template=unsat_template,
            params_schema={
                "properties": {
                    "vars": {"type": "array", "items": {"type": "string"}, "minItems": 1}
                }
            }
        )
        
        # 3. Verify -> Should Fail
        res_fail = runner._verify_macro(gadget)
        assert res_fail.outcome == "FAILED"
        assert any("always UNSAT" in f for f in res_fail.failures)
        assert res_fail.is_satisfiable is False
        
        # 4. Allow UNSAT -> Should Pass
        gadget.meta["allow_unsat"] = True
        res_pass = runner._verify_macro(gadget)
        assert res_pass.outcome == "PASSED"
        assert res_pass.is_satisfiable is False
        
        print("UNSAT rejection logic verified.")

    finally:
        shutil.rmtree(tmp_dir)
