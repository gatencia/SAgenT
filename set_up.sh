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
DENABASE_PATH="Denabase/my_database"
if [ -d "$DENABASE_PATH" ]; then
    echo "Warning: '$DENABASE_PATH' already exists. Skipping initialization."
else
    python Denabase/Denabase/denabase_cli.py init "$DENABASE_PATH"
fi

# 4. Setup Environment Variables
echo "[4/4] Configuring Environment..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "DENABASE_PATH=$DENABASE_PATH" > .env
    echo "# Add your API keys here:" >> .env
    echo "# OPENAI_API_KEY=sk-..." >> .env
    echo "# GOOGLE_API_KEY=AIza..." >> .env
else
    # Append if not exists
    if ! grep -q "DENABASE_PATH" .env; then
        echo "Appending DENABASE_PATH to .env..."
        echo "DENABASE_PATH=$DENABASE_PATH" >> .env
    fi
fi

# 5. Ingest Full Training Set
echo "[5/6] Ingesting SAT-Bench train set into '$DENABASE_PATH'..."
# Using the fast ingestion script with the newly downloaded training data
python scripts/ingest_satbench_fast.py --db "$DE