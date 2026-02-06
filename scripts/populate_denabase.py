#!/usr/bin/env python3
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.ir.ir_types import VarRef, Exactly, AtMost
from Denabase.Denabase.ingest.satbench import load_manifest, ingest_manifest
from Denabase.Denabase.cnf.cnf_io import load_cnf
from Denabase.Denabase.core.errors import CNFError

def main(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Populate Denabase with seeds, manifests, and file scans.")
    parser.add_argument("--db", type=str, required=True, help="Path to database root")
    parser.add_argument("--data", type=str, help="Directory to recursively scan for .cnf files")
    parser.add_argument("--manifest", type=str, help="SAT-Bench JSON/JSONL manifest path")
    parser.add_argument("--root", type=str, help="Root dir for cnf_path fields in manifest")
    parser.add_argument("--family", type=str, help="Default family if scanning --data")
    parser.add_argument("--tags", type=str, help="CSV tags applied to all ingested items")
    parser.add_argument("--verify-ir", action="store_true", help="Verify seeded IR gadgets")
    parser.add_argument("--verify-cnf", action="store_true", help="Verify CNFs ingested from --data scan")
    parser.add_argument("--verify-manifest", action="store_true", help="Verify CNFs ingested from manifest")
    parser.add_argument("--rebuild-indexes", action="store_true", help="Force rebuild structural, NL, and alpha model")
    parser.add_argument("--max-files", type=int, help="Cap number of CNFs scanned from --data")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first ingest failure")

    args = parser.parse_args(args_list)

    db_path = Path(args.db)
    # 1. Initialize DB
    # If db path does not exist or is empty => create it via DenaBase.create
    # Standard DenaBase init handles directory creation, but we follow the request.
    if not db_path.exists() or not any(db_path.iterdir()):
        db = DenaBase.create(str(db_path))
    else:
        db = DenaBase.open(str(db_path))

    summary = {
        "db": str(db_path),
        "seeded": {"ir": 0},
        "manifest": {"ok": 0, "fail": 0},
        "scan": {"ok": 0, "fail": 0},
        "total_entries": 0,
        "indexes": {"struct": False, "nl": False, "alpha": False},
        "failures": []
    }

    tag_list = args.tags.split(",") if args.tags else []

    # 2. Seed IR Gadgets
    try:
        # EXACTLY_ONE on 3 vars
        x1, x2, x3 = VarRef(name="x1"), VarRef(name="x2"), VarRef(name="x3")
        e1 = Exactly(k=1, vars=[x1, x2, x3])
        db.add_ir(e1, "foundation", "exactly_one_3", verify=args.verify_ir, source="seed", tags=tag_list)
        
        # AT_MOST_ONE on 5 vars
        ys = [VarRef(name=f"y{i}") for i in range(1, 6)]
        amo = AtMost(k=1, vars=ys)
        db.add_ir(amo, "foundation", "at_most_one_5", verify=args.verify_ir, source="seed", tags=tag_list)
        
        summary["seeded"]["ir"] = 2
    except Exception as e:
        msg = f"Failed to seed IR gadgets: {e}"
        if args.fail_fast:
            print(json.dumps({"error": msg}))
            sys.exit(1)
        summary["failures"].append({"item": "seeds", "error": msg})

    # 3. Ingest Manifest
    if args.manifest:
        try:
            records = load_manifest(Path(args.manifest))
            manifest_root = Path(args.root) if args.root else None
            
            # Using ingest_manifest helper but we wrap it to handle fail-fast better if needed
            res = ingest_manifest(db, records, root_dir=manifest_root, verify=args.verify_manifest)
            summary["manifest"]["ok"] = res["ok"]
            summary["manifest"]["fail"] = res["fail"]
            summary["failures"].extend(res["failures"])
            
            if args.fail_fast and res["fail"] > 0:
                 print(json.dumps({"error": "Manifest ingestion failed", "details": res["failures"]}))
                 sys.exit(1)
                 
        except Exception as e:
            msg = f"Failed to process manifest: {e}"
            if args.fail_fast:
                print(json.dumps({"error": msg}))
                sys.exit(1)
            summary["failures"].append({"item": "manifest_load", "error": msg})

    # 4. Data Scan
    if args.data:
        data_dir = Path(args.data)
        if not data_dir.exists():
            summary["failures"].append({"item": "data_scan", "error": f"Dir not found: {args.data}"})
        else:
            files = list(data_dir.rglob("*.cnf"))
            if args.max_files:
                files = files[:args.max_files]
            
            for f in files:
                try:
                    doc = load_cnf(f)
                    family = args.family if args.family else f.parent.name
                    if not family or family == ".":
                        family = "unclassified"
                    
                    db.add_cnf(
                        doc, 
                        family=family, 
                        problem_id=f.stem, 
                        verify=args.verify_cnf,
                        source="scan",
                        scan_origin=str(f),
                        tags=tag_list
                    )
                    summary["scan"]["ok"] += 1
                except Exception as e:
                    summary["scan"]["fail"] += 1
                    msg = str(e)
                    summary["failures"].append({"item": str(f), "error": msg})
                    if args.fail_fast:
                        print(json.dumps({"error": f"Failed to ingest {f}: {msg}"}))
                        sys.exit(1)

    # 5. Rebuild Indexes
    if args.rebuild_indexes:
        try:
            db.rebuild_index()
            summary["indexes"]["struct"] = True
        except Exception as e:
            summary["failures"].append({"item": "rebuild_struct", "error": str(e)})

        try:
            db.rebuild_nl_index()
            summary["indexes"]["nl"] = True
        except Exception as e:
            summary["failures"].append({"item": "rebuild_nl", "error": str(e)})

        try:
            db.rebuild_alpha_model()
            summary["indexes"]["alpha"] = True
        except Exception as e:
            summary["failures"].append({"item": "rebuild_alpha", "error": str(e)})

    # Final Stats
    summary["total_entries"] = db.total_entries
    summary["failures"] = summary["failures"][:20] # Cap

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
