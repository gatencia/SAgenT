#!/usr/bin/env python3
"""
Wrapper script to ingest SAT-Bench data using the Denabase CLI logic.
Usage:
    python scripts/ingest_satbench.py --db denabase_storage --manifest satbench.jsonl --root data --verify
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Denabase.Denabase.denabase_cli import handle_ingest_satbench

def main():
    parser = argparse.ArgumentParser(description="Ingest SAT-Bench data into Denabase.")
    parser.add_argument("--db", type=str, required=True, help="Path to database")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSONL")
    parser.add_argument("--root", type=str, help="Root directory for CNF files")
    parser.add_argument("--verify", action="store_true", help="Run strict verification")
    
    args = parser.parse_args()
    
    # Adapt args to match handle_ingest_satbench expectations (db_path instead of db)
    class CLIArgs:
        db_path = args.db
        manifest = args.manifest
        root = args.root
        verify = args.verify
        
    handle_ingest_satbench(CLIArgs())

if __name__ == "__main__":
    main()
