import pytest
from pathlib import Path
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir.ir_types import Exactly, VarRef
from Denabase.Denabase.induce.induce_gadgets import GadgetInducer
from Denabase.Denabase.gadgets.gadget_registry import registry, LearnedGadget

def test_induction_pipeline(tmp_path):
    # Setup DB
    db = DenaBase(str(tmp_path / "db"))
    gadgets_out = tmp_path / "gadgets"
    
    # 1. Seed DB with a known ExactlyOne gadget (n=3)
    # Vars: 1, 2, 3
    ir = Exactly(k=1, vars=[VarRef(name="1"), VarRef(name="2"), VarRef(name="3")])
    eid1 = db.add_ir(ir, family="seed", problem_id="eo_3")
    
    # 2. Run Induction
    inducer = GadgetInducer(db)
    names = inducer.run_induction_pipeline(gadgets_out)
    
    # 3. Assertions
    # Should find one candidate and induce it
    assert len(names) == 1
    name = names[0]
    assert "Induced_ExactlyOne_3" in name
    
    # Check registry
    assert name in registry.list_gadgets()
    g = registry.get(name)
    assert isinstance(g, LearnedGadget)
    assert g.entry_id == eid1
    
    # Check persistence
    assert (gadgets_out / f"{name}.json").exists()
    
    # 4. Test corruption rejection
    # Create a corrupted entry: Identifies as ExactlyOne but isn't
    # We cheat: Add AtMostOne but tag it (via profile modification?) so miner thinks it's ExactlyOne?
    # Or simpler: Add AtMostOne, Miner finds it as AMO, we induce AMO.
    # Then verify AMO logic applies.
    
    # Let's try adding something that looks like AMO profile-wise (k=1)
    # but fails verification (e.g. SAT for weight 2).
    # Hard to forge easily without low-level CNF manipulation.
    
    # Easier: corrupt the CNF of the valid ExactlyOne entry we just added, 
    # then run induction again (on a fresh registry/output to simulate new run).
    
    # Modify CNF to allow all 0 (AtLeastOne violation)
    # This should fail _verify_exactly_one test 2
    
    # Locate CNF
    entry = db.store.get_entry_record(eid1)
    cnf_path = db.root / entry.paths["cnf"]
    original_cnf = cnf_path.read_text()
    
    # Remove the "at least one" clause (1 2 3 0)
    # It's usually the first one or distinct positive clause
    # p cnf 3 4
    # 1 2 3 0 ...
    lines = original_cnf.splitlines()
    # Filter out line with "1 2 3 0" (whitespace variants)
    # Actually just replace it with empty? Or "1 0" logic...
    # Simplest: Just empty the file or make it "p cnf 3 0" -> Tautology -> Satisfies empty set -> fails EO?
    # Tautology satisfies "all 0" (SAT), so verifies AtMostOne(0).
    # But fails ExactlyOne(0) check which expects UNSAT for weight 0.
    
    corrupt_cnf = "p cnf 3 1\n-1 -2 0\n" # Only one pairwise constraint.
    # Weight 0 -> SAT (PASS AMO, FAIL EO)
    # Weight 1 -> SAT (PASS)
    # Weight 2 (1,2) -> UNSAT (PASS)
    # But (1,3) -> SAT (FAIL AMO check 3)
    
    cnf_path.write_text(corrupt_cnf)
    
    # Clear registry memory of previous induction
    prev_len = len(registry.list_gadgets())
    # Note: registry is global, so we can't easily clear it without affecting others?
    # But we can check if it re-registers or throws error.
    # Inducer catches exception.
    
    # Run again
    names_2 = inducer.run_induction_pipeline(gadgets_out)
    
    # Should NOT be in names_2 (or it might fail since ID exists?)
    # Wait, miner will find it again. Inducer will try to verify.
    # Verify should FAIL.
    # So names_2 should be empty (assuming no other candidates).
    
    # Note: names contains the result of THIS run.
    # If verify fails, it won't be appended.
    
    # However, registry might still have the old valid one from step 2?
    # Yes, registry is in-memory.
    # But names_2 return value tells us what was successfully induced *now*.
    
    # Since we corrupted the file on disk, verify() reads disk -> fails -> returns False.
    # So names_2 should be empty.
    
    assert len(names_2) == 0

