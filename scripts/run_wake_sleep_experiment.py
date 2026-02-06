
import argparse
import json
import logging
import random
import time
import os
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Conditional Agent Import (to avoid crash if LLM not set up)
try:
    from engine.agent import ReActAgent
    from engine.solution.types import SatStatus
except ImportError:
    ReActAgent = None

from Denabase.Denabase.sleep.sleep_runner import SleepRunner
from Denabase.Denabase.db.denabase import DenaBase
from Denabase.agent.denabase_bridge import DenabaseBridge
from Denabase.Denabase.trace import EncodingTrace, TraceEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wake_sleep_experiment")

def generate_synthetic_trace(problem_id: str, family: str) -> EncodingTrace:
    """Generates a synthetic trace with repeated motifs for testing miner."""
    # Motif: Chain of Implications (A->B, B->C)
    events = []
    
    # Per problem, generate 3 chains
    for i in range(3):
        base = i * 10
        # Chain: x -> y -> z
        # Vars: x=v{base}, y=v{base+1}, z=v{base+2}
        v_x = f"v{base}"
        v_y = f"v{base+1}"
        v_z = f"v{base+2}"
        
        # event 1: Implies(x, y)
        events.append(TraceEvent(kind="IR_NODE", payload={
            "type": "implies", "a": v_x, "b": v_y, "arity": 2, "vars": [v_x, v_y]
        }))
        
        # event 2: Implies(y, z)
        events.append(TraceEvent(kind="IR_NODE", payload={
            "type": "implies", "a": v_y, "b": v_z, "arity": 2, "vars": [v_y, v_z]
        }))
        
    return EncodingTrace(problem_id=problem_id, family=family, events=events)

# ... imports ...
from engine.llm_provider import make_llm

# ... (generate_synthetic_trace remains) ...

def main():
    parser = argparse.ArgumentParser(description="Run Wake-Sleep Experiment")
    parser.add_argument("--db", required=True, help="Path to Denabase")
    parser.add_argument("--manifest", required=True, help="Path to problem manifest JSON")
    parser.add_argument("--root", default="experiment_results", help="Root dir for results")
    parser.add_argument("--n", type=int, default=10, help="Number of problems to run per epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sleep", action="store_true", help="Enable Sleep phase")
    parser.add_argument("--repeat", type=int, default=1, help="Number of Wake/Sleep epochs")
    parser.add_argument("--synthetic-wake", action="store_true", help="Generate synthetic traces instead of running Agent")
    parser.add_argument("--provider", type=str, default="google", choices=["openai", "google", "ollama", "mock"], help="LLM Provider")
    parser.add_argument("--model", type=str, help="LLM Model")
    parser.add_argument("--api_key", type=str, help="LLM API Key")
    
    args = parser.parse_args()
    
    # Load .env manually if needed (for keys)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v.strip().strip('"')

    run_experiment(args)

def run_experiment(args):
    db_path = Path(args.db)
    manifest_path = Path(args.manifest)
    results_root = Path(args.root)
    results_root.mkdir(exist_ok=True, parents=True)
    
    # Init DB if needed
    if not db_path.exists():
         DenaBase.create(db_path)
    
    db = DenaBase.open(str(db_path))
    bridge = DenabaseBridge.get_instance(str(db_path))

    # Load Manifest
    full_problems = []
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            try:
                # Try reading as standard JSON list
                full_problems = json.load(f)
            except json.JSONDecodeError:
                # Fallback to JSONL
                f.seek(0)
                try:
                    full_problems = [json.loads(line) for line in f if line.strip()]
                except Exception as e:
                    logger.warning(f"Failed to parse manifest as JSON or JSONL: {e}")
            except Exception as e:
                logger.warning(f"Error reading manifest: {e}")
    
    if not full_problems:
        logger.warning(f"Manifest {manifest_path} empty or missing or invalid.")
        if args.synthetic_wake:
             logger.info("Generating dummy problems for synthetic wake.")
             full_problems = [{"id": f"synthetic_prob_{i}", "goal": f"Solve synthetic problem {i}"} for i in range(args.n)]
        else:
             logger.error("Cannot run real agent without problems.")
             return

    random.seed(args.seed)
    logger.info(f"Starting Wake-Sleep Experiment. Epochs: {args.repeat}. Sleep: {args.sleep}")
    
    for epoch in range(args.repeat):
        epoch_id = f"epoch_{epoch}"
        logger.info(f"=== Starting {epoch_id} ===")
        
        # Reload Gadgets (Wake Phase Prep)
        bridge.reload_gadgets()
        
        # Sample problems
        problems = random.sample(full_problems, min(args.n, len(full_problems)))
        
        if args.synthetic_wake:
            logger.info("Running SYNTHETIC Wake Phase (Mocking traces)...")
            from Denabase.Denabase.cnf.cnf_types import CnfDocument
            
            for p in problems:
                pid = p.get("id", "unknown")
                # Create dummy entry
                entry_id = db.add_cnf(CnfDocument(clauses=[], num_vars=0), "synthetic", pid, verify=False)
                
                # Attach synthetic trace
                trace = generate_synthetic_trace(pid, "synthetic")
                db.attach_trace(entry_id, trace)
                logger.info(f"Attached synthetic trace to {pid}")
                
        else:
            # REAL AGENT RUN
            if ReActAgent is None:
                logger.error("Agent engine not available. Install engine or use --synthetic-wake.")
                return

            logger.info(f"Running REAL Agent Wake Phase using {args.provider}...")
            
            try:
                llm = make_llm(args.provider, args.api_key, args.model)
            except Exception as e:
                logger.error(f"Failed to create LLM: {e}")
                return

            agent = ReActAgent(llm_callable=llm)
            
            for p_idx, prob in enumerate(problems):
                # Map ID
                pid = prob.get("id") or prob.get("_denabase_row_id") or "unknown"
                logger.info(f"Solving problem {p_idx+1}/{len(problems)}: {pid}")
                
                # Map Goal
                goal = prob.get("goal") or prob.get("description") or prob.get("natural_language")
                
                # Handle SATBench format (Scenario + Conditions + Question)
                if not goal and "scenario" in prob:
                    parts = [prob["scenario"]]
                    if "conditions" in prob:
                        conds = prob["conditions"]
                        if isinstance(conds, list):
                            parts.append("Conditions:\n" + "\n".join(conds))
                        else:
                            parts.append(f"Conditions: {conds}")
                    if "question" in prob:
                        parts.append(f"Question: {prob['question']}")
                    
                    goal = "\n\n".join(parts)
                
                if not goal:
                     if "data" in prob:
                          goal = f"Solve this problem: {json.dumps(prob['data'])}"
                     else:
                          logger.warning("Skipping problem without goal")
                          continue
                
                logger.info(f"Goal: {goal[:100]}...")
                try:
                    state = agent.run(goal)
                    logger.info(f"Result: {state.final_status}")
                except Exception as e:
                    logger.error(f"Agent failed on problem: {e}")

        # Sleep Phase
        if args.sleep:
            logger.info("Entering Sleep Phase...")
            runner = SleepRunner(db)
            try:
                res = runner.run_sleep_cycle(min_freq=2, top_k=5)
                logger.info(f"Sleep Cycle Result: {json.dumps(res, indent=2)}")
            except Exception as e:
                logger.error(f"Sleep cycle failed: {e}", exc_info=True)
 
    logger.info("Experiment Completed.")

if __name__ == "__main__":
    main()
