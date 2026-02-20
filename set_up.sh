#!/bin/bash
# Denabase Setup Script
# This script handles both host-side Docker building and container-side initialization.

set -e

# Detect if we are already inside the container
INSIDE_DOCKER=false
if [ -f /.dockerenv ] || [ "$IS_CONTAINER" = "true" ]; then
    INSIDE_DOCKER=true
fi

if [ "$INSIDE_DOCKER" = "false" ]; then
    echo "=== Denabase Host Setup (Docker) ==="
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed. Please install Docker to proceed."
        exit 1
    fi

    # 1. Ensure local directories and .env exist for volume mounting
    echo "[1/3] Preparing local workspace..."
    mkdir -p data outputs models logs tmp
    
    if [ ! -f .env ]; then
        echo "Initializing .env from .env.example..."
        cp .env.example .env
    fi

    # 2. Interactive API Key Setup
    if ! grep -q "API_KEY=AIza" .env && ! grep -q "API_KEY=sk-" .env; then
        echo ""
        echo "--- LLM API Configuration ---"
        echo "No API keys detected in your .env file."
        echo "Which provider would you like to use? (google/openai/none)"
        read -p "Provider: " PROVIDER
        
        if [ "$PROVIDER" = "google" ]; then
            read -p "Enter your Google Gemini API Key: " API_KEY
            sed -i.bak "s/GOOGLE_API_KEY=.*/GOOGLE_API_KEY=$API_KEY/" .env && rm .env.bak
        elif [ "$PROVIDER" = "openai" ]; then
            read -p "Enter your OpenAI API Key: " API_KEY
            sed -i.bak "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$API_KEY/" .env && rm .env.bak
        fi
        echo "Configuration saved to .env"
    fi

    # 3. Build the Docker image
    echo "[2/3] Building Docker image..."
    docker compose build

    # 3. Run the internal setup inside the container
    echo "[3/3] Running internal setup inside container..."
    docker compose run --rm -e IS_CONTAINER=true app bash ./set_up.sh
    
    echo ""
    echo "=== Full Host Setup Complete! ==="
    echo "You can now run benchmarks using Docker:"
    echo "docker compose run --rm app python bench/run_bench.py --id mrpp_8x8_4r_T20_sat_central_block --provider simulated"
    exit 0
fi

# --- INTERNAL CONTAINER SETUP LOGIC ---
echo "=== Denabase Internal Container Setup ==="

# Export PYTHONPATH to ensure scripts can import Denabase
export PYTHONPATH=$PYTHONPATH:.

# Set local cache directories
CACHE_DIR="${CACHE_DIR:-tmp}"
mkdir -p "$CACHE_DIR/hf_home" "$CACHE_DIR/pip_cache"
export HF_HOME="$(pwd)/$CACHE_DIR/hf_home"
export PIP_CACHE_DIR="$(pwd)/$CACHE_DIR/pip_cache"

# 1. Install Dependencies
echo "[1/4] Installing Denabase and dependencies..."
python -m pip install -e '.[test]' || true

# 2. Download SAT-Bench
echo "[2/4] Downloading SAT-Bench from Hugging Face..."
python Denabase/data/download_satbench.py

# 3. Initialize Database
echo "[3/4] Initializing database..."
MODELS_DIR="${MODELS_DIR:-Denabase}"
DENABASE_PATH="$MODELS_DIR/my_database"
if [ -d "$DENABASE_PATH" ]; then
    echo "Warning: '$DENABASE_PATH' already exists. Skipping initialization."
else
    python Denabase/Denabase/denabase_cli.py init "$DENABASE_PATH"
fi

# 4. Ingest Full Training Set
echo "[4/4] Ingesting SAT-Bench train set..."
python scripts/ingest_satbench_fast.py --db "$DENABASE_PATH" --manifest Denabase/data/satbench/satbench_train.jsonl

# 5. Repair & Visualize
echo "[5/5] Ensuring Database Integrity..."
python scripts/repair_denabase.py --db "$DENABASE_PATH"

echo ""
echo "=== Internal Setup Complete! ==="
