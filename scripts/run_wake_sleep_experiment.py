
import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Any

from engine.agent import ReActAgent
from engine.solution.types import SatStatus
from Denabase.sleep.sleep_runner import SleepRunner
from Denabase.db.denabase import DenaBase
from agent.denabase_bridge import DenabaseBridge


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wake_sleep_experiment")

def main():
    parser = argparse.ArgumentParser(description="Run Wake-Sleep Experiment")
    parser.add_argument("--db", required=True, help="Path to Denabase")
    parser.add_argument("--manifest", required=True, help="Path to problem manifest JSON")
    parser.add_argument("--root", default="experiment_results", help="Root dir for results")
    parser.add_argument("--n", type=int, default=10, help="Number of problems to run per epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sleep", action="store_true", help="Enable Sleep phase")
    parser.add_argument("--repeat", type=int, default=2, help="Number of Wake/Sleep epochs")
    
    args = parser.parse_args()
    
    run_experiment(args)

def run_experiment(args):
    db_path = Path(args.db)
    manifest_path = Path(args.manifest)
    results_root = Path(args.root)
    results_root.mkdir(exist_ok=True, parents=True)
    
    # Init DB if needed (SleepRunner needs it)
    if not db_path.exists():
         DenaBase.create(db_path)
         
    # Load Manifest
    with open(manifest_path, "r") as f:
        full_problems = json.load(f)
        
    random.seed(args.seed)
    
    logger.info(f"Starting Wake-Sleep Experiment. Epochs: {args.repeat}. Sleep: {args.sleep}")
    
    experiment_log = []
    
    for epoch in range(args.repeat):
        epoch_id = f"epoch_{epoch}"
        logger.info(f"=== Starting {epoch_id} ===")
        
        # 1. Wake Phase
        # Reload Gadgets
        DenabaseBridge.get_instance(str(db_path)).reload_gadgets()
        
        # Select problems
        problems = random.sample(full_problems, min(args.n, len(full_problems)))
        epoch_metrics = {
            "epoch": epoch,
            "solved_count": 0,
            "total_time": 0,
            "results": []
        }
        
        for p_idx, prob in enumerate(problems):
            logger.info(f"Solving problem {p_idx+1}/{len(problems)}: {prob.get('id', 'unknown')}")
            
            # Mock Agent or Real? Real.
            # We assume manifest has 'goal' text.
            goal = prob.get("goal") or prob.get("description")
            if not goal:
                logger.warning("Skipping problem without goal")
                continue
                
            # Init Agent
            # NOTE: Agent config might need tweaking to force specific backend? 
            # Default is usually PB or Minizinc.
            # We assume default config is fine.
            # We define a dummy LLM? No, we need REAL LLM for real experiment.
            # But for testing, we might mock. 
            # The script assumes environment is set up for Agent (API keys etc).
            
            # If we are in this environment, we might not have API keys set.
            # The user requested a "Reproducible Experiment Harness".
            # This implies using the actual agent.
            
            agent = ReActAgent()
            state = agent.run(goal)
            
            # Metrics
            is_sat = (state.sat_result and state.sat_result.status == SatStatus.SAT)
            time_taken = state.sat_result.time_taken if state.sat_result else 0
            
            res_entry = {
                "problem_id": prob.get("id"),
                "status": state.sat_result.status if state.sat_result else "UNKNOWN",
                "time_ms": time_taken,
                "steps": state.step_count,
                "encoding_size": len(state.cnf_clauses) if state.cnf_clauses else 0,
                # "retrieval_success": ... # Hard to measure without ground truth
            }
            epoch_metrics["results"].append(res_entry)
            
            if is_sat:
                epoch_metrics["solved_count"] += 1
                epoch_metrics["total_time"] += time_taken
                
            # Note: Trace is auto-captured by Agent->Bridge
            
        experiment_log.append(epoch_metrics)
        
        # Save intermediate results
        with open(results_root / "experiment_log.json", "w") as f:
            json.dump(experiment_log, f, indent=2)
            
        # 2. Sleep Phase
        if args.sleep and epoch < args.repeat - 1:
            logger.info("Entering Sleep Phase...")
            runner = SleepRunner(str(db_path))
            # Run cycle (Mine -> Verify -> Pack)
            # We assume default config for induction
            try:
                pack_path = runner.run_cycle()
                if pack_path:
                    logger.info(f"Sleep produced new pack: {pack_path}")
                else:
                    logger.info("Sleep produced no new gadgets.")
            except Exception as e:
                logger.error(f"Sleep cycle failed: {e}")

    logger.info("Experiment Completed.")

if __name__ == "__main__":
    main()
