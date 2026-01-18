# SAT ReAct Benchmark Harness

This directory contains the benchmark infrastructure for the SAT ReAct agent.

## Structure
- `instances/`: JSON benchmark problems.
- `checkers/`: Python scripts to validate solutions.
- `runs/`: Output logs of agent runs.
- `run_bench.py`: CLI execution script.

## Adding New Families
1. **Instances**: Add JSON files to `instances/`.
   - Must have "family": "your_family_name".
   - Must have "expected" field.
2. **Checker**: Add `checkers/your_family_name.py`.
   - Must implement `def check(solution, instance) -> (bool, list[str])`.

## Running Benchmarks
Run all instances:
```bash
python3 bench/run_bench.py
```

Filter by family:
```bash
python3 bench/run_bench.py --family mrpp
```

Filter by ID:
```bash
python3 bench/run_bench.py --id mrpp_6x6_3r_T8
```

## Supported Families
- **MRPP**: Multi-Robot Path Planning.
  - Solution schema: `{"paths": {robot_name: [[x,y], ...]}}`
