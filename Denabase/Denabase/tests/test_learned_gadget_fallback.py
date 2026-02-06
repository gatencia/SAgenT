
import pytest
import shutil
import tempfile
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.gadgets.gadget_registry import registry, LearnedGadget
from Denabase.Denabase.ir.ir_compile import compile_ir, CompilationContext, encode_fixed_cnf
from Denabase.Denabase.ir.ir_types import FixedCNF, VarRef, Lit
from Denabase.Denabase.cnf.cnf_types import CnfDocument

def test_learned_gadget_fallback():
    # 1. Setup Temp DB
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "db"
    
    try:
        db = DenaBase.create(db_path)
        # Inject DB into registry
        registry.set_db(db)
        
        # 2. Create a CNF entry (AtMostOne of 3 vars: 1, 2, 3)
        # Clauses: (-1 -2), (-1 -3), (-2 -3)
        # Internal Vars: 1, 2, 3
        clauses = [[-1, -2], [-1, -3], [-2, -3]] 
        doc = CnfDocument(clauses=clauses, num_vars=3)
        
        entry_id = db.add_cnf(doc, "test_fam", "test_p", verify=False)
        
        # 3. Create LearnedGadget
        # It's an "unknown" gadget, but we know it takes 3 vars
        # inferred_type = "Unknown"
        lg = LearnedGadget(
            name="MysteryGadget",
            entry_id=entry_id,
            params={"inferred_type": "Unknown", "arity": 3}
        )
        registry.register_learned(lg)
        
        # 4. Use the gadget in a new compilation
        # We want to use it on variables ["x", "y", "z"]
        # So internal 1->x, 2->y, 3->z
        ir = lg.build_ir(params={"vars": ["x", "y", "z"]})
        
        # 5. Assert it returned FixedCNF
        assert isinstance(ir, FixedCNF)
        assert len(ir.clauses) == 3
        # Vars check
        mapped_vars = [v.name for v in ir.vars]
        assert mapped_vars == ["x", "y", "z"]
        
        # 6. Compile to final CNF
        # Also add a constraint that x is True, to force others to be False
        # And y is True -> Expect UNSAT if everything mapped correctly?
        # Wait, AMO(x,y,z) means at most one is True.
        # If x is True, y and z must be False.
        
        # Let's just key compilation output first
        clauses, varmap = compile_ir([ir])
        
        # Check mapping
        # x->1, y->2, z->3 (sorted)
        assert varmap["x"] == 1
        assert varmap["y"] == 2
        assert varmap["z"] == 3
        
        # Clauses should be: (-1 -2), (-1 -3), (-2 -3)
        # Order might vary
        c_set = set(tuple(sorted(c)) for c in clauses)
        expected = {(-2, -1), (-3, -1), (-3, -2)}
        assert c_set == expected
        
        print("Fallback compilation verified!")

    finally:
        shutil.rmtree(tmp_dir)
        registry.set_db(None)
        if "MysteryGadget" in registry._learned_registry:
            del registry._learned_registry["MysteryGadget"]

def test_fixed_cnf_with_aux():
    # Test FixedCNF that has internal auxiliaries
    # Input vars: 2 (internal 1, 2)
    # Clauses: 1 -> 3, 2 -> 3, 3 -> False (meaning 1 and 2 must be false)
    # 3 is internal aux
    
    ctx = CompilationContext(["a", "b"]) # a->1, b->2
    # Current next_aux = 3
    
    fixed_ir = FixedCNF(
        clauses=[[-1, 3], [-2, 3], [-3]],
        vars=[VarRef(name="a"), VarRef(name="b")]
    )
    
    encode_fixed_cnf(fixed_ir, ctx)
    
    # Internal var 3 from FixedCNF should be mapped to a new aux in ctx
    # ctx.allocate_aux returns 3
    # So mapping: 1->1(a), 2->2(b), 3->3(new_aux)
    # Clauses: (-1, 3), (-2, 3), (-3)
    
    assert len(ctx.clauses) == 3
    # Verify new aux allocated
    assert ctx.next_aux == 4
    
