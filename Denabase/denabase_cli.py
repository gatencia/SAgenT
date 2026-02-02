import argparse
import sys
import json
from pathlib import Path
from typing import Any, List, Union

from Denabase.db.denabase import DenaBase
from Denabase.cnf.cnf_types import CnfDocument
from Denabase.cnf.cnf_io import load_cnf
from Denabase.core.errors import DenabaseError
from Denabase.ir.ir_types import IR

def load_ir_file(path: Path) -> Any:
    """Loads and validates an IR JSON file."""
    try:
        data = json.loads(path.read_text())
        # IR model validation
        ir_model = IR.model_validate(data)
        return ir_model.root
    except Exception as e:
        raise DenabaseError(f"Invalid IR file: {e}")

def handle_init(args):
    print(f"Initializing Denabase at {args.path}...")
    db = DenaBase.create(args.path)
    print("Done.")

def handle_add_cnf(args):
    print(f"Adding CNF {args.cnf_file} to {args.db_path} (verify={args.verify})...")
    meta = json.loads(args.meta) if args.meta else {}
    db = DenaBase.open(args.db_path)
    doc = load_cnf(Path(args.cnf_file))
    entry_id = db.add_cnf(doc, args.family, args.problem_id, meta, verify=args.verify)
    print(f"Successfully added {args.problem_id} (ID: {entry_id}).")

def handle_add_ir(args):
    print(f"Adding IR {args.ir_file} to {args.db_path} (verify={args.verify})...")
    meta = json.loads(args.meta) if args.meta else {}
    db = DenaBase.open(args.db_path)
    ir_obj = load_ir_file(Path(args.ir_file))
    entry_id = db.add_ir(ir_obj, args.family, args.problem_id, meta, verify=args.verify)
    print(f"Successfully added {args.problem_id} (ID: {entry_id}).")

def handle_query(args):
    db = DenaBase.open(args.db_path)
    q_file = Path(args.query_file) if args.query_file else None
    
    # Try loading as CNF first, then fallback to IR
    query_obj = None
    if q_file:
        try:
            query_obj = load_cnf(q_file)
        except Exception:
            try:
                query_obj = load_ir_file(q_file)
            except Exception as e:
                raise DenabaseError(f"Query file must be valid CNF or IR: {e}")
        
    alpha = float(args.alpha) if args.alpha else None
    use_learned = not args.no_learned_alpha
    
    results = db.query_similar(
        query_obj, 
        topk=args.topk, 
        alpha=alpha, 
        nl_query_text=args.nl_text,
        use_learned_alpha=use_learned
    )
    
    if not results:
        print("No matches found.")
    else:
        for i, res in enumerate(results):
            print(f"{i+1}. {res['problem_id']} (ID: {res['entry_id']})")
            print(f"   Family: {res['family']} | Dedupe Key: {res['dedupe_key']}")
            print(f"   Score: {res['final_score']:.4f} [Struct: {res['structural_similarity']:.4f}, NL: {res['nl_similarity']:.4f}, Alpha: {res['alpha_used']:.2f}]")

def handle_ingest_satbench(args):
    from Denabase.ingest.satbench import load_manifest, ingest_manifest
    db = DenaBase.open(args.db_path)
    records = load_manifest(args.manifest)
    print(f"Ingesting {len(records)} records...")
    ok, fail = ingest_manifest(db, records, root=args.root, verify=args.verify)
    print(f"Done. Success: {ok}, Failures: {fail}")

def main():
    parser = argparse.ArgumentParser(description="Denabase CLI - Manage verified SAT encodings.")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a new Denabase.")
    init_parser.add_argument("path", type=str, help="Path to create the database directory.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing DB if present.")
    init_parser.set_defaults(func=handle_init)

    # add-cnf
    add_parser = subparsers.add_parser("add-cnf", help="Add a CNF to Denabase.")
    add_parser.add_argument("db_path", type=str, help="Path to the Denabase directory.")
    add_parser.add_argument("cnf_file", type=str, help="Path to the .cnf file.")
    add_parser.add_argument("--family", type=str, required=True, help="Problem family.")
    add_parser.add_argument("--problem-id", type=str, required=True, help="Specific ID.")
    add_parser.add_argument("--meta", type=str, help="Optional JSON string with metadata.")
    add_parser.add_argument("--verify", action="store_true", help="Run strict verification (can be slow).")
    add_parser.set_defaults(func=handle_add_cnf)

    # add-ir
    add_ir_parser = subparsers.add_parser("add-ir", help="Add an IR definition to Denabase.")
    add_ir_parser.add_argument("db_path", type=str, help="Path to the Denabase directory.")
    add_ir_parser.add_argument("ir_file", type=str, help="Path to the IR JSON file.")
    add_ir_parser.add_argument("--family", type=str, required=True, help="Problem family.")
    add_ir_parser.add_argument("--problem-id", type=str, required=True, help="Specific ID.")
    add_ir_parser.add_argument("--meta", type=str, help="Optional JSON string with metadata.")
    add_ir_parser.add_argument("--verify", action="store_true", help="Run strict verification (can be slow).")
    add_ir_parser.set_defaults(func=handle_add_ir)

    # Query
    query_parser = subparsers.add_parser("query", help="Query similar CNFs")
    query_parser.add_argument("db_path", type=str, help="Path to database")
    query_parser.add_argument("--query-file", type=str, help="CNF or IR file to query with")
    # Legacy positional support for convenience if needed, but making it optional is easier with -- flag
    query_parser.add_argument("--topk", type=int, default=5, help="Number of results")
    query_parser.add_argument("--alpha", type=str, help="Override structural trust alpha (0.0-1.0)")
    query_parser.add_argument("--nl-text", type=str, help="Natural language query text")
    query_parser.add_argument("--no-learned-alpha", action="store_true", help="Disable learned alpha model")
    query_parser.set_defaults(func=handle_query)
    
    # Ingest SAT-Bench
    ingest_parser = subparsers.add_parser("ingest-satbench", help="Ingest SAT-Bench dataset")
    ingest_parser.add_argument("db_path", type=str, help="Path to database")
    ingest_parser.add_argument("--manifest", type=str, required=True, help="Path to JSON/JSONL manifest")
    ingest_parser.add_argument("--root", type=str, help="Root directory for CNF files")
    ingest_parser.add_argument("--verify", action="store_true", help="Run strict verification")
    ingest_parser.set_defaults(func=handle_ingest_satbench)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
