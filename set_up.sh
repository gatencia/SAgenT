#!/bin/bash
# Denabase Setup Script
# Run this once to initialize the environment and database.

set -e

echo "=== Denabase Setup ==="

# Export PYTHONPATH to ensure scripts can import Denabase even if pip install fails
export PYTHONPATH=$PYTHONPATH:.

# Set local cache directories to avoid permission errors
mkdir -p tmp/hf_home tmp/pip_cache
export HF_HOME=$(pwd)/tmp/hf_home
export PIP_CACHE_DIR=$(pwd)/tmp/pip_cache

# 1. Install Dependencies
echo "[1/4] Installing Denabase and dependencies..."
# Try to install dependencies, but don't fail the script if it permissions errors (common in restricted envs)
python -m pip install -e '.[test]' || echo "Warning: Pip install failed. Continuing with existing environment..."

# 2. Download SAT-Bench
echo "[2/4] Downloading SAT-Bench from Hugging Face..."
# This requires 'datasets' which was added to pyproject.toml
python Denabase/data/download_satbench.py

# 3. Initialize Database
echo "[3/4] Initializing 'Denabase/my_database'..."
if [ -d "Denabase/my_database" ]; then
    echo "Warning: 'Denabase/my_database' already exists. Skipping initialization."
else
    python Denabase/Denabase/denabase_cli.py init Denabase/my_database
fi

# 4. Ingest Full Training Set
echo "[4/4] Ingesting SAT-Bench train set into 'Denabase/my_database'..."
# Using the fast ingestion script with the newly downloaded training data
python scripts/ingest_satbench_fast.py --db Denabase/my_database --manifest Denabase/data/satbench/satbench_train.jsonl

# 5. Generate Knowledge Graph
echo "[5/5] Generating Knowledge Graph..."
python scripts/visualize_db.py --db Denabase/my_database --nodes 600 --edges 2 || echo "Warning: Visualization failed."

echo ""
echo "=== Setup Complete! ==="
echo "You can now explore the knowledge graph: check 'denabase_graph.html'"
echo "Or run queries like:"
echo "python Denabase/Denabase/denabase_cli.py query Denabase/my_database --nl-text 'logical puzzle'"
