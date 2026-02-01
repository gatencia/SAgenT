import argparse
import sys
import json
from pathlib import Path
from Denabase.db.denabase import DenaBase

def main():
    parser = argparse.ArgumentParser(description="Denabase CLI - Manage verified SAT encodings.")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a new Denabase.")
    init_parser.add_argument("path", type=str, help="Path to create the database directory.")

    # add-cnf
    add_parser = subparsers.add_parser("add-cnf", help="Add a CNF to Denabase.")
    add_parser.add_argument("db_path", type=str, help="Path to the Denabase directory.")
    add_parser.add_argument("cnf_file", type=str, help="Path to the .cnf file.")
    add_parser.add_argument("--family", type=str, required=True, help="Problem family (e.g., crypto, scheduling).")
    add_parser.add_argument("--problem-id", type=str, required=True, help="Specific ID for the problem instance.")
    add_parser.add_argument("--meta", type=str, help="Optional JSON string with metadata.")

    # query-cnf
    query_parser = subparsers.add_parser("query-cnf", help="Query similar CNFs.")
    query_parser.add_argument("db_path", type=str, help="Path to the Denabase directory.")
    query_parser.add_argument("cnf_file", type=str, help="Path to the .cnf file to query.")
    query_parser.add_argument("--topk", type=int, default=5, help="Number of results to return.")

    args = parser.parse_args()

    if args.command == "init":
        print(f"Initializing Denabase at {args.path}...")
        db = DenaBase(args.path)
        print("Done.")

    elif args.command == "add-cnf":
        print(f"Adding CNF {args.cnf_file} to {args.db_path}...")
        meta = json.loads(args.meta) if args.meta else {}
        db = DenaBase(args.db_path)
        entry_id = db.add_cnf(args.cnf_file, args.family, args.problem_id, meta)
        print(f"Successfully added {args.problem_id} (ID: {entry_id}).")

    elif args.command == "query-cnf":
        print(f"Querying similar CNFs for {args.cnf_file} in {args.db_path} (topk={args.topk})...")
        db = DenaBase(args.db_path)
        results = db.query_similar(args.cnf_file, topk=args.topk)
        
        if not results:
            print("No matches found.")
        else:
            for i, res in enumerate(results):
                print(f"{i+1}. {res.metadata.problem_id} (Family: {res.metadata.family})")
                print(f"   Variables: {res.stats.n_vars}, Clauses: {res.stats.n_clauses}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
