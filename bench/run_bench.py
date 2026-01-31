import sys
import os
import json
import argparse
import importlib.util
import time
import urllib.request
import urllib.error
import ssl
import ssl
from typing import Dict, Any, Callable

# Add parent directory to path to import react_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from engine import ReActAgent, AgentState
except ImportError:
    print("Error: Could not import ReActAgent from engine package")
    sys.exit(1)

# Manual .env loading to avoid dependencies
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    os.environ[k] = v
                except: pass

def load_instances(instances_dir: str, family_filter: str = None, id_filter: str = None):
    instances = []
    if not os.path.exists(instances_dir):
        return []
    
    for filename in os.listdir(instances_dir):
        if filename.endswith(".json"):
            path = os.path.join(instances_dir, filename)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    if family_filter and data.get("family") != family_filter:
                        continue
                    if id_filter and data.get("id") != id_filter:
                        continue
                    instances.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {filename}")
    return sorted(instances, key=lambda x: x.get("id", ""))

def load_checker(family: str, checkers_dir: str):
    # Mapping for families where checker filename differs
    mapping = {
        "polyomino": "pentomino"
    }
    filename = mapping.get(family, family)
    checker_path = os.path.join(checkers_dir, f"{filename}.py")
    if not os.path.exists(checker_path):
        return None
    
    spec = importlib.util.spec_from_file_location(f"checker_{family}", checker_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "check"):
        return module.check
    return None

# ==============================================================================
# LLM PROVIDERS
# ==============================================================================

class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last_call = 0.0
    
    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

def call_openai_api(prompt: str, api_key: str, model: str = "gpt-4-turbo-preview", insecure: bool = False) -> str:
    # Use urllib for zero dependencies
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    
    # SSL Context
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            res_body = response.read()
            res_json = json.loads(res_body)
            return res_json["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI API Error: {e.read().decode()}")

def call_google_api(prompt: str, api_key: str, model: str = "gemini-2.0-flash", insecure: bool = False) -> str:
    # Google Generative Language API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    
    # SSL Context
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    
    max_retries = 5
    base_wait = 2.0
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                res_body = response.read()
                res_json = json.loads(res_body)
                # Parse Gemini response structure
                try:
                    return res_json["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    return json.dumps(res_json) # Return full debug if structure fails
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = base_wait * (2 ** attempt)
                print(f"Rate limited (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Google API Error: {e.read().decode()}")
        except Exception as e:
            # Handle other transient errors
            wait_time = base_wait * (2 ** attempt)
            print(f"API Error ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue
            
    raise RuntimeError("Google API Retry Limit Exceeded")

def call_ollama_api(prompt: str, model: str = "llama3") -> str:
    # Default Ollama local URL
    url = "http://localhost:11434/api/chat"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json"
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read()
            res_json = json.loads(res_body)
            return res_json["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama API Error: {e.read().decode()}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama Connection Refused: {e}. Is Ollama running on port 11434?")

def call_mock_llm(prompt: str) -> str:
    """Mock that just quits."""
    return json.dumps({
        "thought": "I am a mock LLM. I cannot solve this, but I will terminate.",
        "action": "FINISH",
        "action_input": {}
    })

def call_simulated_llm(prompt: str) -> str:
    """
    Simulated 'Perfect' Agent for mrpp_6x6_3r_T8.
    It inspects the prompt (history) to decide the next step.
    This validates the harness logic without needing an API key or Solver.
    """
    # 1. Check history in prompt to see what step we are at
    # The prompt contains: "...HISTORY:\n0. T:thought A:... O:..."
    
    # Heuristic: count lines starting with digit + "."
    step = 0
    lines = prompt.splitlines()
    for l in lines:
        if len(l) > 2 and l[0].isdigit() and l[1] == '.' and "T:" in l:
            try:
                sid = int(l.split('.')[0])
                step = max(step, sid + 1)
            except: pass
            
    # Policy for mrpp_6x6_3r_T8 (Phased)
    if step == 0:
        return json.dumps({
            "thought": "Start in Observation phase. Update plan.",
            "action": "UPDATE_PLAN",
            "action_input": {
                "plan": "1. Define vars\n2. Add constraints\n3. Solve", 
                "problem_notes": "We formulate the Multi-Robot Path Planning task as a decision problem. Let R be a set of robots on a G=(V,E) grid graph (2x2). The goal is to move each robot r from start to goal in T=2 timesteps without collisions.",
                "observations": ["Board is 2x2 grid", "2 Robots involved", "Time Horizon T=2"],
                "variables": [
                    "pos_R0_x_y_t: Boolean variable indicating if Robot 0 is at (x,y) at time t.",
                    "pos_R1_x_y_t: Boolean variable indicating if Robot 1 is at (x,y) at time t."
                ],
                "strategy": "We employ a Time-Expanded Graph encoding. We create boolean variables for every valid (robot, time, location) tuple. Constraints ensure: 1) Each robot is at exactly one place per time step. 2) Moves follow grid adjacency. 3) No vertex or edge collisions allowed.",
                "verification": "CONFIRMED"
            }
        })
    elif step == 1:
        return json.dumps({
            "thought": "Advance to Variables phase.",
            "action": "ADVANCE_PHASE", 
            "action_input": {}
        })
    elif step == 2:
        return json.dumps({
            "thought": "Define variables.",
            "action": "DEFINE_VARIABLES",
            "action_input": ["pos_R0_0_0_t0", "pos_R1_1_1_t2"] 
        })
    elif step == 3:
        return json.dumps({
            "thought": "Advance to Constraints phase.",
            "action": "ADVANCE_PHASE",
            "action_input": {}
        })
    elif step == 4:
        # Mini MRPP (2x2 Grid, 2 Robots, T=2)
        # R0 starts at (0,0) needs (1,1)
        # R1 starts at (1,1) needs (0,0)
        # Simple swap around corners logic with tiny CNF
        code = r"""
        % Variables: pos_R{r}_{x}_{y}_t{t}
        var bool: pos_R0_0_0_t0; var bool: pos_R0_0_1_t1; var bool: pos_R0_1_1_t2;
        var bool: pos_R1_1_1_t0; var bool: pos_R1_1_0_t1; var bool: pos_R1_0_0_t2;
        
        % Start/Goal conditions
        constraint pos_R0_0_0_t0 = true; 
        constraint pos_R0_1_1_t2 = true;
        constraint pos_R1_1_1_t0 = true; 
        constraint pos_R1_0_0_t2 = true;
        
        % Valid Transition: (0,0)->(0,1) for R0
        constraint pos_R0_0_0_t0 -> pos_R0_0_1_t1;
        constraint pos_R0_0_1_t1 -> pos_R0_1_1_t2;
        
        % Valid Transition: (1,1)->(1,0) for R1
        constraint pos_R1_1_1_t0 -> pos_R1_1_0_t1;
        constraint pos_R1_1_0_t1 -> pos_R1_0_0_t2;
        
        % Helper Not Condition
        constraint not (pos_R0_0_1_t1 /\ pos_R1_1_0_t1 = false);
        """
        return json.dumps({
            "thought": "Add Mini MRPP Code.",
            "action": "ADD_MINIZINC_CODE",
            "action_input": code
        })
    elif step == 5:
        return json.dumps({"thought": "Solve the model.", "action": "SOLVE", "action_input": {}})
    elif step == 6:
        # Construct the report manually as the Agent
        report = """=== FINAL SAT REPORT ===

1. PROBLEM FORMULATION
----------------------
We formulate the Multi-Robot Path Planning task as a decision problem. Let R be a set of robots on a G=(V,E) grid graph (2x2). The goal is to move each robot r from start to goal in T=2 timesteps without collisions.

2. VARIABLE DEFINITIONS
----------------------
- pos_R0_x0_y0_t0: Robot 0 at (0,0) at t=0
- pos_R0_x0_y1_t1: Robot 0 at (0,1) at t=1
- pos_R0_x1_y1_t2: Robot 0 at (1,1) at t=2
(And similarly for Robot 1)

3. LOGICAL CONSTRAINTS & STRATEGY
---------------------------------
We employ a Time-Expanded Graph encoding. We create boolean variables for every valid (robot, time, location) tuple. 
Constraints ensure: 
1) Each robot is at exactly one place per time step. 
2) Moves follow grid adjacency. 
3) No vertex or edge collisions allowed.

4. OBSERVATIONS
---------------
- Board is 2x2 grid
- 2 Robots involved
- Time Horizon T=2

5. SATISFIABILITY RESULT
------------------------
SAT

=== SOLUTION ===
Robot 0 Path: (0,0)->(0,1)->(1,1)
Robot 1 Path: (1,1)->(1,0)->(0,0)

=== VARIABLE LEGEND ===
(Generated from SOLVE output)

=== FULL CNF CLAUSES (Human Readable) ===
(pos_R0_0_0_t0)
(pos_R0_1_1_t2)
(pos_R1_1_1_t0)
(pos_R1_0_0_t2)
(NOT pos_R0_0_0_t0 OR pos_R0_0_1_t1)
(NOT pos_R0_0_1_t1 OR pos_R0_1_1_t2)
... (and so on)
"""
        return json.dumps({"thought": "Finish and write report.", "action": "FINISH", "action_input": {"report": report}})
    
    return json.dumps({"action": "FINISH", "action_input": {}})

def make_llm(provider: str, api_key: str = None, model: str = None, insecure: bool = False) -> Callable[[str], str]:
    if provider == "openai":
        if not api_key: raise ValueError("API Key required for openai provider")
        m = model if model else "gpt-4-turbo-preview"
        return lambda p: call_openai_api(p, api_key, m, insecure=insecure)
    elif provider == "google":
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("API Key required for google provider (arg or GOOGLE_API_KEY env)")
        m = model if model else "gemini-2.5-flash"
        # Gemini Free Tier is often ~15 RPM => 4 seconds per request
        limiter = RateLimiter(min_interval=4.0)
        def limited_call(p):
            limiter.wait()
            return call_google_api(p, key, m, insecure=insecure)
        return limited_call
    elif provider == "ollama":
        m = model if model else "llama3"
        return lambda p: call_ollama_api(p, m)
    elif provider == "simulated":
        return call_simulated_llm
    else:
        return call_mock_llm


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_benchmark(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(base_dir, "instances")
    checkers_dir = os.path.join(base_dir, "checkers")
    runs_dir = os.path.join(base_dir, "runs")
    
    # Cleanup old artifacts
    if os.path.exists("output.txt"): os.remove("output.txt")
    if os.path.exists("output_debug.json"): os.remove("output_debug.json")

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    instances = load_instances(instances_dir, args.family, args.id)
    print(f"Found {len(instances)} instances to run.")
    
    results = []

    print(f"{'ID':<25} | {'Family':<10} | {'Status':<10} | {'Checker':<10} | {'Steps':<5} | {'Constrs':<5}")
    print("-" * 80)

    for instance in instances:
        instance_id = instance["id"]
        family = instance["family"]
        
        # 1. Build validation checker
        checker_fn = load_checker(family, checkers_dir)
        
        # 2. Construct Goal
        goal = (
            f"Solve this instance and return a decoded JSON solution.\n"
            f"Problem: {instance.get('natural_language', '')}\n"
            f"Data: {json.dumps(instance['data'])}\n"
            f"Please ensure you set state.solution to a JSON object matching the problem requirements."
        )

        # 3. Initialize Agent
        try:
            llm = make_llm(args.provider, args.api_key, args.model, insecure=args.insecure)
            agent = ReActAgent(llm_callable=llm, max_steps=args.max_steps, ir_backend=args.IR)
        except Exception as e:
            print(f"Error initializing agent: {e}")
            sys.exit(1)
        
        # 4. Run Agent
        start_time = time.time()
        try:
            state = agent.run(goal)
        except Exception as e:
            print(f"Agent crashed on {instance_id}: {e}")
            # Log partial
            duration = time.time() - start_time
            # Create a dummy state for logging if needed
            continue # or fake state?
            
        duration = time.time() - start_time
        
        # 5. Check Solution
        checker_ok = False
        checker_errors = []
        expected_sat = instance.get("expected", {}).get("sat", True)
        agent_sat_status = getattr(state.sat_result, "status", None) if state.sat_result else None
        
        if expected_sat:
            # Case 1: Expected SAT
            if state.solution and checker_fn:
                checker_ok, checker_errors = checker_fn(state.solution, instance)
            elif not state.solution:
                checker_ok = False
                checker_errors = ["Expected SAT but got no solution in state.solution"]
            else:
                checker_ok = True # No checker, but solution exists
        else:
            # Case 2: Expected UNSAT
            from engine.solution.types import SatStatus
            if agent_sat_status == SatStatus.UNSAT:
                checker_ok = True
                checker_errors = ["Correctly identified UNSAT."]
            else:
                checker_ok = False
                checker_errors = [f"Expected UNSAT but agent result was {agent_sat_status}"]
        
        # 6. Log Results
        run_data = {
            "instance_id": instance_id,
            "family": family,
            "timestamp": time.time(),
            "duration_seconds": duration,
            "final_status": state.final_status,
            "checker_ok": checker_ok,
            "checker_errors": checker_errors,
            "steps_count": len(state.trajectory),
            "constraints_count": len(state.model_constraints),
            "trajectory": state.trajectory, # might be large
            "model_constraints": [
                {
                    "id": c.id, 
                    "backend": c.ir_backend, 
                    "kind": c.kind, 
                    "params": c.parameters
                } for c in state.model_constraints
            ],
            "solution": state.solution,
            "metrics": state.metrics if hasattr(state, "metrics") else {}
        }
        
        run_file = os.path.join(runs_dir, f"{instance_id}.run.json")
        with open(run_file, 'w') as f:
            json.dump(run_data, f, indent=2)
            
        # 7. Print Summary
        status_str = str(state.final_status) if state.final_status else "None"
        checker_str = "PASS" if checker_ok else "FAIL"
        print(f"{instance_id:<25} | {family:<10} | {status_str:<10} | {checker_str:<10} | {len(state.trajectory):<5} | {len(state.model_constraints):<5}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAT ReAct Benchmark Runner")
    parser.add_argument("--family", type=str, help="Filter by family")
    parser.add_argument("--id", type=str, help="Filter by instance ID")
    parser.add_argument("--max_steps", type=int, default=20, help="Max agent steps")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "openai", "google", "ollama", "simulated"], help="LLM Provider")
    parser.add_argument("--api_key", type=str, help="API Key for Provider")
    parser.add_argument("--model", type=str, help="Model name (e.g. gemini-1.5-pro-latest)")
    parser.add_argument("--IR", type=str, default="pb", help="Intermediate Representation Backend (pb, minizinc, etc)")
    parser.add_argument("--insecure", action="store_true", help="Bypass SSL certificate verification (macOS workaround)")
    
    args = parser.parse_args()
    run_benchmark(args)
