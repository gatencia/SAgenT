#!/bin/bash
# Denabase Setup Script
# Run this once to initialize the environment and database.

set -e

echo "=== Denabase Setup ==="

# 1. Install Dependencies
echo "[1/4] Installing Denabase and dependencies..."
pip install -e '.[test]'

# 2. Download SAT-Bench
echo "[2/4] Downloading SAT-Bench from Hugging Face..."
# This requires 'datasets' which was added to pyproject.toml
python data/download_satbench.py

# 3. Initialize Database
echo "[3/4] Initializing 'my_database'..."
if [ -d "my_database" ]; then
    echo "Warning: 'my_database' already exists. Skipping initialization."
else
    python Denabase/denabase_cli.py init my_database
fi

# 4. Ingest Full Training Set
echo "[4/4] Ingesting SAT-Bench train set into 'my_database'..."
# Using the fast ingestion script with the newly downloaded training data
python scripts/ingest_satbench_fast.py --db my_database --manifest data/satbench/satbench_train.jsonl

echo ""
echo "=== Setup Complete! ==="
echo "You can now run queries like:"
echo "python Denabase/denabase_cli.py query my_database --nl-text 'logical puzzle'"
