# Agentic SAT Solver (LLM-SAT)

A **ReAct-based Agent** that uses Large Language Models (LLMs) to reason about problems and a **SAT Solver** (`python-sat`) to rigorously solve them.

This project demonstrates a "Brain-Muscle" architecture:
-   **Brain**: Gemini 2.0 (or local Ollama models) translates natural language goals into mathematical constraints.
-   **Muscle**: Python-SAT compiles these constraints into CNF and finds the exact solution (or proves UNSAT).
-   **Safety**: Explicit **Fuzzing** steps allow the agent to unit-test its own logic before solving.

## Features
*   **Zero-Dependency LLM Calls**: Uses standard `urllib` for API interactions (Gemini, OpenAI, Ollama).
*   **JIT Compilation**: Translates high-level constraints (e.g., `at_most_k`) to CNF on the fly.
*   **Deterministic Fuzzing**: Validates agent logic probabilistically.
*   **Benchmark Harness**: Includes validaters (Checkers) for Multi-Robot Path Planning (MRPP).

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
