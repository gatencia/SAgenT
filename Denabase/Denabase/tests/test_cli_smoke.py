import pytest
import subprocess
import json
import sys
from pathlib import Path

import os

# Helper to run CLI
def run_cli(args):
    cmd = [sys.executable, "Denabase/denabase_cli.py"] + args
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    return subprocess.run(cmd, env=env, capture_output=True, text=True)

def test_cli_smoke(tmp_path):
    db_path = tmp_path / "test_db"
    
    # 1. Init
    res = run_cli(["init", str(db_path)])
    if res.returncode != 0:
        print(f"CLI init failed. stderr:\n{res.stderr}")
    assert res.returncode == 0
    assert "Initializing Denabase" in res.stdout
    assert db_path.exists()
    
    # 2. Add CNF
    cnf_file = tmp_path / "test.cnf"
    cnf_file.write_text("p cnf 2 2\n1 2 0\n-1 -2 0\n")
    
    res = run_cli([
        "add-cnf", 
        str(db_path), 
        str(cnf_file), 
        "--family", "smoke", 
        "--problem-id", "prob_cnf"
    ])
    assert res.returncode == 0
    assert "Successfully added prob_cnf" in res.stdout

    # 2b. Add CNF with Verify
    res = run_cli([
        "add-cnf", 
        str(db_path), 
        str(cnf_file), 
        "--family", "smoke", 
        "--problem-id", "prob_cnf_verified",
        "--verify"
    ])
    assert res.returncode == 0
    assert "Successfully added prob_cnf_verified" in res.stdout
    
    # 3. Add IR
    ir_file = tmp_path / "test.json"
    # IR JSON: {"kind": "exactly", "k": 1, "vars": [{"name": "a"}, {"name": "b"}]}
    ir_data = {
        "kind": "exactly",
        "k": 1, 
        "vars": [{"name": "a"}, {"name": "b"}]
    }
    ir_file.write_text(json.dumps(ir_data))
    
    res = run_cli([
        "add-ir",
        str(db_path),
        str(ir_file),
        "--family", "smoke",
        "--problem-id", "prob_ir"
    ])
    assert res.returncode == 0
    assert "Successfully added prob_ir" in res.stdout
    
    # 4. Query (using IR)
    res = run_cli([
        "query",
        str(db_path),
        str(ir_file),
        "--topk", "5"
    ])
    assert res.returncode == 0
    assert "prob_ir" in res.stdout
    # prob_cnf might also appear if similarity is handled loosely (it's small)
    
    # 5. Query (using CNF)
    res = run_cli([
        "query",
        str(db_path),
        str(cnf_file)
    ])
    assert res.returncode == 0
    assert "prob_cnf" in res.stdout
