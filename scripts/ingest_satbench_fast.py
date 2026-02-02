import argparse
import sys
import json
import random
from pathlib import Path
from tqdm import tqdm
from Denabase.db.denabase import DenaBase
from Denabase.ingest.satbench import load_manifest

def main():
    parser = argparse.ArgumentParser(description="Fast SAT-Bench ingestion for Denabase.")
    parser.add_argument("--db", required=True, help="Path to database")
    parser.add_argument("--manifest", required=True, help="Path to JSON/JSONL manifest")
    parser.add_argument("--root", help="Root directory for CNF files")
    parser.add_argument("--verify-sample-rate", type=float, default=0.0, help="Fraction of items to verify (0.0-1.0)")
    parser.add_argument("--verify-max", type=int, default=0, help="Max items to verify")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling determinism")
    parser.add_argument("--no-rebuild", action="store_true", help="Skip index rebuild at end")
    
    args = parser.parse_args()
    
    # 1. Load Manifest
    records = load_manifest(args.manifest)
    print(f"Loaded {len(records)} records from manifest.")
    
    # 2. Determine verification indices
    random.seed(args.seed)
    indices = list(range(len(records)))
    
    to_verify = set()
    if args.verify_sample_rate > 0:
        n = int(len(records) * args.verify_sample_rate)
        sampled = random.sample(indices, min(n, len(records)))
        to_verify.update(sampled)
        
    if args.verify_max > 0:
        # If already sampled some, maybe replenish or just limit
        count = max(0, args.verify_max - len(to_verify))
        if count > 0:
            remaining = [i for i in indices if i not in to_verify]
            sampled = random.sample(remaining, min(count, len(remaining)))
            to_verify.update(sampled)

    print(f"Verification: {len(to_verify)} items selected for spot-check.")

    # 3. Bulk Ingest
    from Denabase.cnf.cnf_types import CnfDocument
    db = DenaBase.open(args.db)
    
    ok, fail = 0, 0
    
    with db.bulk_ingest(rebuild=not args.no_rebuild):
        for i, rec in enumerate(tqdm(records, desc="Ingesting")):
            should_verify = (i in to_verify)
            
            try:
                # Resolve IDs and basic info
                did = rec.get("id") or rec.get("dataset_id") or rec.get("problem_id") or rec.get("_denabase_row_id") or f"unnamed_{i}"
                family = rec.get("family", "unknown")
                nl_text = rec.get("natural_language") or rec.get("scenario") or rec.get("description") or ""
                label = rec.get("label")
                if label is None:
                    label = rec.get("satisfiable")
                split = rec.get("split")
                
                # Resolve CNF
                cnf_path = rec.get("cnf_path")
                cnf_dimacs = rec.get("cnf_dimacs")
                clauses = rec.get("clauses")
                cnf_doc = None
                
                if clauses is not None:
                    nv = rec.get("num_vars")
                    if nv is None:
                        nv = 0
                        for c in clauses:
                            for lit in c:
                                nv = max(nv, abs(lit))
                    cnf_doc = CnfDocument(num_vars=nv, clauses=clauses)
                
                if cnf_path and args.root:
                    p = Path(args.root) / cnf_path
                    if p.exists():
                        cnf_path = str(p)
                
                db.add_satbench_case(
                    dataset_id=did,
                    family=family,
                    problem_id=did,
                    nl_text=nl_text,
                    expected_label=label,
                    split=split,
                    cnf_path=cnf_path,
                    cnf_dimacs=cnf_dimacs,
                    cnf_doc=cnf_doc,
                    tags=rec.get("tags", []),
                    verify=should_verify
                )
                ok += 1
            except Exception as e:
                print(f"\nFailed {rec.get('problem_id')}: {e}")
                fail += 1
                
    print(f"\nIngestion Complete. Success: {ok}, Failures: {fail}")
    if not args.no_rebuild:
        print("Indexes rebuilt successfully.")

if __name__ == "__main__":
    main()
