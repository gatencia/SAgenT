#!/usr/bin/env python3
"""
Download SATBench dataset from Hugging Face and write to disk as JSONL.

Outputs:
  data/satbench/
    manifest.json
    satbench_<split>.jsonl
    satbench_sample.jsonl

Usage:
  python data/download_satbench.py
Optional env vars:
  SATBENCH_DATASET (default: "LLM4Code/SATBench")
  SATBENCH_OUTDIR  (default: "<repo>/data/satbench")
  SATBENCH_SAMPLE_N (default: 50)
"""

from __future__ import annotations

import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

def _stable_id(row: Dict[str, Any], fallback_idx: int) -> str:
    """
    Try to construct a stable id from common fields. If none exist,
    hash the canonical JSON of the row.
    """
    for k in ("id", "problem_id", "uid", "name"):
        if k in row and row[k] is not None and str(row[k]).strip():
            return str(row[k])
    blob = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()[:16]
    return f"row_{fallback_idx}_{h}"

def _jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def _dataset_to_rows(ds) -> List[Dict[str, Any]]:
    # Convert HF dataset rows to plain python dicts
    rows: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        r = dict(ds[i])
        r["_denabase_row_id"] = _stable_id(r, i)
        rows.append(r)
    return rows

def _pick_sample(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """
    Deterministic sample: sort by stable id then take first n.
    """
    rows_sorted = sorted(rows, key=lambda x: x.get("_denabase_row_id", ""))
    return rows_sorted[: min(n, len(rows_sorted))]

def main() -> int:
    dataset_name = os.environ.get("SATBENCH_DATASET", "LLM4Code/SATBench")
    outdir_env = os.environ.get("SATBENCH_OUTDIR", "")
    sample_n = int(os.environ.get("SATBENCH_SAMPLE_N", "50"))

    # Resolve output directory: default <repo>/data/satbench relative to this file
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    outdir = Path(outdir_env).expanduser().resolve() if outdir_env else (repo_root / "data" / "satbench")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except Exception as e:
        print("ERROR: Missing dependency 'datasets'. Install with: pip install datasets", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2

    print(f"[SATBench] Loading dataset: {dataset_name}")
    ds_dict = load_dataset(dataset_name)

    # ds_dict is usually a DatasetDict with splits, but could be a single Dataset
    splits: List[Tuple[str, Any]] = []
    if hasattr(ds_dict, "keys"):
        # DatasetDict-like
        for split_name in ds_dict.keys():
            splits.append((split_name, ds_dict[split_name]))
    else:
        splits.append(("data", ds_dict))

    manifest: Dict[str, Any] = {
        "dataset": dataset_name,
        "outdir": str(outdir),
        "splits": {},
    }

    all_rows_for_sample: List[Dict[str, Any]] = []

    for split_name, ds in splits:
        print(f"[SATBench] Processing split '{split_name}' with {len(ds)} rows...")
        rows = _dataset_to_rows(ds)
        all_rows_for_sample.extend(rows)

        # Track columns
        columns = list(rows[0].keys()) if rows else []
        manifest["splits"][split_name] = {
            "rows": len(rows),
            "columns": columns,
        }

        # Write split file
        split_path = outdir / f"satbench_{split_name}.jsonl"
        n_written = _jsonl_write(split_path, rows)
        print(f"[SATBench] Wrote {n_written} rows -> {split_path}")

    # Deterministic sample across all splits
    sample_rows = _pick_sample(all_rows_for_sample, sample_n)
    sample_path = outdir / "satbench_sample.jsonl"
    _jsonl_write(sample_path, sample_rows)
    print(f"[SATBench] Wrote deterministic sample ({len(sample_rows)} rows) -> {sample_path}")

    # Save manifest
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[SATBench] Wrote manifest -> {manifest_path}")

    print("[SATBench] Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())