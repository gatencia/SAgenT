import sys
import argparse
from pathlib import Path

# Fix import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Denabase.Denabase.db.denabase import DenaBase

def main():
    parser = argparse.ArgumentParser(description="Repair Denabase Index")
    parser.add_argument("--db", required=True, help="Path to database")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    print(f"Opening DB at {db_path}...")
    db = DenaBase.open(str(db_path))
    
    print("Rebuilding NL Index...")
    db.rebuild_nl_index()
    print("Done Rebuilding NL Index.")
    
    print("Rebuilding Structural Index...")
    db.rebuild_index()
    print("Done Rebuilding Structural Index.")
    
    print("Repair Complete.")

if __name__ == "__main__":
    main()
