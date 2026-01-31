# LLM-SAT: Agentic Constraint Solving

LLM-SAT is a **ReAct-based Agentic Framework** that bridges the gap between high-level natural language reasoning and low-level SAT/Constraint solving. It uses Large Language Models (LLMs) to formulate problems and rigorous solvers (PySAT, MiniZinc) to solve them.

---

## 🚀 How it Works: The 5-Phase Lifecycle

The agent operates in a strictly phased environment, enforced by the **SATManager** "referee". This ensures the model is built incrementally and correctly.

### 1. 🔍 OBSERVATION (Discovery)
The agent analyzes the problem description to extract grid sizes, piece counts, blocked cells, and movement rules.
- **Action**: `UPDATE_PLAN`
- **Goal**: Populate `PLAN.md` with enough observations to advance.

### 2. 🔢 VARIABLES (State Space)
The agent defines a discrete search space by mapping observations to Boolean or Integer variables.
- **Actions**: `DEFINE_VARIABLES`, `DEFINE_VARIABLE_PATTERN`
- **Output**: A registered variable registry (e.g., `pos_robot1_t0_x0_y1`).

### 3. ⚖️ CONSTRAINTS (Logic Formulation)
The agent translates mathematical or logical rules into high-level IR (Internal Representation) constraints.
- **Actions**: `ADD_MODEL_CONSTRAINTS`, `ADD_PYTHON_CONSTRAINT_BLOCK`
- **Kinds**: `exactly_one`, `at_most_k`, `implies`, `all_different`, etc.
- **Verification**: `FUZZ_CONSTRAINTS` or `TEST_CONSTRAINT` can be used to "locally" verify logic before a full solve.

### 4. ⚡ IMPLEMENTATION (JIT Compilation & Solving)
The `SATManager` JIT-compiles the high-level model into low-level instructions (CNF, Pseudo-Boolean, or FlatZinc) and executes the backend solver.
- **Action**: `SOLVE`
- **Backends**: `pb` (Pseudo-Boolean), `cnf` (Raw SAT), `minizinc` (Constraint Programming).

### 5. 🛠 REFINEMENT (Debugging)
If the solver returns `UNSAT` (unexpectedly) or fails validation, the agent analyzes the failure, critiques its model, and iterates.
- **Action**: `REFINE_FROM_VALIDATION`, `REMOVE_MODEL_CONSTRAINTS`

---

## 🛠 Action Reference

| Action | Description |
| :--- | :--- |
| **`UPDATE_PLAN`** | Update `PLAN.md` with observations and goals. |
| **`ADVANCE_PHASE`** | Transition to the next stage of the lifecycle. |
| **`DEFINE_VARIABLES`** | Register individual Boolean variables. |
| **`DEFINE_VARIABLE_PATTERN`**| Register variable families (e.g., `grid_{row}_{col}`). |
| **`ADD_MODEL_CONSTRAINTS`** | Add high-level logic (ExactlyOne, AtMostK, etc.). |
| **`ADD_PYTHON_CONSTRAINT_BLOCK`** | **Power User**: Execute Python code to generate complex constraints programmatically. |
| **`FUZZ_CONSTRAINTS`** | Generate random test cases to verify the JIT compiler's output. |
| **`TEST_CONSTRAINT`** | Verify a specific constraint against a manual truth-table entry. |
| **`SOLVE`** | Compile and run the backend solver. |
| **`DECODE_SOLUTION`** | Map raw boolean assignments back to domain variables. |
| **`FINISH`** | Complete the task and provide the final solution. |

---

## 🏗 Project Structure

- **`engine/`**: The core logic.
    - `agent.py`: The ReAct loop and tool-calling coordinator.
    - `sat_manager.py`: The "Referee" and JIT compiler entry point.
    - `backends/`: Transformation logic for `pb`, `cnf`, and `minizinc`.
    - `vars.py`: Global variable registry and ID mapper.
- **`bench/`**: Benchmark harness.
    - `run_bench.py`: CLI to run specific instances or families.
    - `instances/`: JSON-encoded problems (GraphColoring, Pentomino, MRPP).
    - `checkers/`: Domain-specific Python validators for solutions.
- **`memory/`**: Workspace for intermediate solver files (e.g., `.fzn`, `.opb`).

---

## 🔧 Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires `python-sat` and optionally `minizinc` installed on your PATH.*

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Add your GOOGLE_API_KEY (for Gemini) or use Ollama for local runs.
   ```

3. **Run a Benchmark**:
   ```bash
   # Run a specific problem instance
   python3 bench/run_bench.py --id poly_pentomino_10x6_hole_sat --provider google
   ```

---

## 🛡 Verification & Fuzzing

The repo includes a built-in **Fuzzing Engine** to prevent "modeling errors" from wasting solver time. When the agent adds a complex Python block, it can call `FUZZ_CONSTRAINTS` to generate 50+ random assignments, checking if the solver's result (SAT/UNSAT) matches the expected logic for that specific constraint subset.

---

> [!IMPORTANT]
> **Deterministic Phasing**: The agent CANNOT skip to `SOLVE` without passing through `VARIABLES` and `CONSTRAINTS`. The `SATManager` will reject out-of-order actions.
