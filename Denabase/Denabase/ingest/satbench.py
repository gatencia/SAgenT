import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.Denabase.cnf.cnf_types import CnfDocument

logger = logging.getLogger(__name__)

def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """
    Loads records from JSON or JSONL.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
        
    records = []
    try:
        if path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            # Assume JSON array or single object
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    if "splits" in data and isinstance(data["splits"], dict):
                        # Flatten all splits into a single list
                        for split_name, split_records in data["splits"].items():
                            if isinstance(split_records, list):
                                for r in split_records:
                                    if isinstance(r, dict):
                                        r["split"] = split_name
                                records.extend(split_records)
                    else:
                        records = [data]
                else:
                    records = [data]
    except Exception as e:
        raise ValueError(f"Failed to parse manifest {path}: {e}")
        
    return records

def ingest_manifest(db: DenaBase, 
                    records: List[Dict[str, Any]], 
                    root_dir: Path = None, 
                    verify: bool = False) -> Dict[str, Any]:
    """
    Ingests a list of SAT-Bench records into Denabase.
    """
    ok_count = 0
    fail_count = 0
    failures = []
    
    total = len(records)
    print(f"Ingesting {total} records...")
    
    for i, rec in enumerate(records):
        try:
            # Extract basic fields
            # Required: id, family, natural_language, label
            # Optional: cnf_path, cnf_dimacs
            
            did = rec.get("id") or rec.get("problem_id") or rec.get("_denabase_row_id") or str(i)
            family = rec.get("family", "unknown")
            nl_text = rec.get("natural_language") or rec.get("scenario") or rec.get("description") or ""
            if not nl_text:
                logger.warning(f"Record {did} has missing/empty natural language text.")
            label = rec.get("label")
            if label is None:
                label = rec.get("satisfiable") # Handle boolean satisfiable
            split = rec.get("split")
            
            cnf_path_raw = rec.get("cnf_path")
            cnf_dimacs = rec.get("cnf_dimacs")
            clauses = rec.get("clauses")
            
            # Resolve path if needed
            cnf_path = None
            if cnf_path_raw:
                if root_dir:
                    p = root_dir / cnf_path_raw
                else:
                    p = Path(cnf_path_raw)
                cnf_path = str(p)

            cnf_doc = None
            if clauses is not None:
                nv = rec.get("num_vars")
                if nv is None:
                    # Infer num_vars from clauses
                    nv = 0
                    for c in clauses:
                        for lit in c:
                            nv = max(nv, abs(lit))
                cnf_doc = CnfDocument(num_vars=nv, clauses=clauses)
                
            if not cnf_path and not cnf_dimacs and not cnf_doc:
                raise ValueError("Record missing 'cnf_path', 'cnf_dimacs' or 'clauses'")
                
            # Ingest
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
                user_meta=rec,
                verify=verify
            )
            ok_count += 1
            
        except Exception as e:
            fail_count += 1
            msg = str(e)
            failures.append({"id": rec.get("id", f"idx_{i}"), "error": msg})
            # Log but continue
            # print(f"Failed record {i}: {msg}")
            
    return {
        "total": total,
        "ok": ok_count,
        "fail": fail_count,
        "failures": failures[:20] # Cap output
    }
