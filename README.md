# Agentic SAT Solver (LLM-SAT)

A **ReAct-based Agent** that uses Large Language Models (LLMs) to reason about problems and a **SAT Solver** (`python-sat`) to rigorously solve them.

## 🏗 System Architecture (Strict Phasing)

The agent now follows a strict **5-Phase Pipeline** to ensure robust problem formulation:

1.  **OBSERVATION** (Understanding):
    *   Agent extracts bounds, grid sizes, and rules.
    *   Output: `UPDATE_PLAN` with rich observations.
    *   **Goal**: Must satisfy observation count before advancing.

2.  **VARIABLES** (Search Space):
    *   Agent maps observations to discrete Boolean/Integer variables.
    *   Action: `DEFINE_VARIABLES`.
    *   **Goal**: Must register variables in the engine.

3.  **CONSTRAINTS** (Logic):
    *   Agent applies high-level constraints (e.g., `at_most_k`, `alldifferent`) to variables.
    *   Action: `ADD_MODEL_CONSTRAINTS`.
    *   **Goal**: Must populate the model with logic.

4.  **IMPLEMENTATION** (Solving):
    *   Agent calls `SOLVE`.
    *   The backend (PySAT or MiniZinc) compiles and runs the solver.

5.  **DEBUGGING** (Refinement):
    *   If solving fails, the agent enters this loop to critique and fix the model.

### Key Features
*   **Deterministic Phase Advancement**: The Engine acts as a referee, explicitly telling the Agent when it has met the criteria to `ADVANCE_PHASE`.
*   **Multi-Backend**: Support for `minizinc`, `pb` (Pseudo-Boolean), and `cnf`.
*   **Iterative Planning**: `PLAN.md` is updated live, preserving "Current Code" visibility.

## Installation

1.  Clone the repository.
2.  Install the core dependency:
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires `python-sat`)*

## Configuration

Create a `.env` file in the root directory for your API keys:

```bash
# .env
GOOGLE_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here (optional)
```

## Usage

### 1. Run Benchmarks (Google Gemini)
To run the agent on a specific Multi-Robot Path Planning instance:

```bash
python3 bench/run_bench.py --provider google --id mrpp_6x6_3r_T8
```
*   **Note**: Includes automatic rate-limiting (4s delay) for the Free Tier.

### 2. Run Benchmarks (Local Ollama)
Run completely offline with local models (requires [Ollama](https://ollama.com/) running):

```bash
python3 bench/run_bench.py --provider ollama --model llama3 --id mrpp_6x6_3r_T8
```

## Project Structure

*   `react_engine.py`: The core Agent, SAT Manager, and JIT Compiler.
*   `bench/`: Benchmark harness.
    *   `run_bench.py`: CLI Runner.
    *   `instances/`: JSON problem files.
    *   `checkers/`: Domain-specific solution validators.
    *   `runs/`: Output logs.

## License
MIT
