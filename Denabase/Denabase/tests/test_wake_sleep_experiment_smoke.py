
import pytest
import shutil
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.run_wake_sleep_experiment import run_experiment
from engine.agent import ReActAgent
from engine.state import AgentState
from engine.solution.types import SatResult, SatStatus

class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def test_wake_sleep_smoke():
    tmp_dir = tempfile.mkdtemp()
    try:
        root = Path(tmp_dir)
        db_path = root / "test_db"
        manifest_path = root / "manifest.json"
        
        # 1. Create Hacky Manifest
        problems = [
            {"id": "p1", "goal": "Solve problem 1"},
            {"id": "p2", "goal": "Solve problem 2"}
        ]
        with open(manifest_path, "w") as f:
            json.dump(problems, f)
            
        # 2. Mock Agent to avoid LLM calls
        # We need to mock ReActAgent used inside the script
        with patch("scripts.run_wake_sleep_experiment.ReActAgent") as MockAgentClass:
            mock_agent_instance = MagicMock()
            MockAgentClass.return_value = mock_agent_instance
            
            # Outcome: Solved
            mock_state = AgentState()
            mock_state.sat_result = SatResult(status=SatStatus.SAT, time_taken=100.0)
            mock_state.cnf_clauses = [[1, 2], [-1, -2]] # Fake encoding
            
            # The agent.run returns this state
            mock_agent_instance.run.return_value = mock_state
            
            # 3. Mock SleepRunner to avoid expensive mining
            with patch("scripts.run_wake_sleep_experiment.SleepRunner") as MockSleep:
                mock_runner = MagicMock()
                MockSleep.return_value = mock_runner
                mock_runner.run_cycle.return_value = Path("dummy/pack")
                
                # 4. Run Experiment
                args = MockArgs(
                    db=str(db_path),
                    manifest=str(manifest_path),
                    root=str(root / "results"),
                    n=2,
                    seed=123,
                    sleep=True,
                    repeat=2
                )
                
                run_experiment(args)
                
                # 5. Assertions
                # Check metrics log
                log_path = root / "results" / "experiment_log.json"
                assert log_path.exists()
                with open(log_path) as f:
                    log = json.load(f)
                    
                assert len(log) == 2 # 2 epochs
                assert log[0]["epoch"] == 0
                assert log[0]["solved_count"] == 2 # All mock solved
                
                # Check Sleep Called
                assert mock_runner.run_cycle.call_count == 1 # Only between epoch 0 and 1
            
    finally:
        shutil.rmtree(tmp_dir)
