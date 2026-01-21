import json
from typing import Callable, Optional, Tuple, Dict, Any, List

from engine.state import AgentState
from engine.config import IRConfig
from engine.actions import ActionType
from engine.backends.registry import IRBackendRegistry
from engine.sat_manager import SATManager

class ReActAgent:
    def __init__(self, 
                 llm_callable: Callable[[str], str], 
                 max_steps: int = 10,
                 validator_callable: Optional[Callable[[Dict[str, Any], AgentState], Tuple[bool, List[str]]]] = None):
        self.llm_callable = llm_callable
        self.max_steps = max_steps
        self.validator_callable = validator_callable
        self.registry = IRBackendRegistry()
        self.sat = SATManager(self.registry)
        self.config = IRConfig.from_env_or_file() 
        
        # Load System Prompt
        import os
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "system.txt")
        with open(prompt_path, "r") as f:
            self.system_prompt = f.read()

    def run(self, goal: str) -> AgentState:
        state = AgentState()
        state.active_ir_backend = self.config.backend 
        while not state.finished and state.step_count < self.max_steps:
            state.step_count += 1
            print(f"Step {state.step_count}...", end="\r", flush=True)
            history_prompt = self._construct_prompt(goal, state)
            raw = self.llm_callable(history_prompt)
            try:
                resp = self._parse_llm_output(raw)
            except Exception as e:
                self._record(state, "ParseFail", "ERROR", f"{e} || RAW: {raw}")
                continue
            
            action_type = resp["action"]
            action_input = resp["action_input"]

            # Execute Action
            obs = self._execute(state, action_type, action_input)
            
            # Post-Execution Validation Hook
            if action_type == ActionType.DECODE_SOLUTION and state.solution and self.validator_callable:
                is_valid, errs = self.validator_callable(state.solution, state)
                state.validator_results.append({"pass": is_valid, "errors": errs})
                if not is_valid:
                     # Force Refine Policy
                     obs = f"Solution Found but INVALID. Errors: {errs}. You MUST use REFINE_FROM_VALIDATION to fix."
                     state.final_status = "INVALID" # Prevent early finish
                else:
                     obs = "Solution Verified VALID."
            
            self._record(state, resp.get("thought",""), f"{action_type.value}", obs)
            if action_type == ActionType.FINISH: state.finished = True
        return state

    def _execute(self, state: AgentState, action: ActionType, arg: Any) -> str:
        try:
            if action == ActionType.DEFINE_VARIABLES: return self.sat.define_variables(state, arg)
            elif action == ActionType.CREATE_PLAN: return self.sat.create_plan(state, arg)
            elif action == ActionType.REFINE_FROM_VALIDATION: return self.sat.refine_from_validation(state, arg.get("errors", []))
            elif action == ActionType.ADD_MODEL_CONSTRAINTS: return self.sat.add_model_constraints(state, arg)
            elif action == ActionType.REMOVE_MODEL_CONSTRAINTS: return self.sat.remove_model_constraints(state, arg)
            elif action == ActionType.LIST_IR_SCHEMA: return self.sat.get_schema(state)
            elif action == ActionType.SOLVE: return self.sat.solve(state)
            elif action == ActionType.TEST_CONSTRAINT: return self.sat.test_constraint(state, arg)
            elif action == ActionType.FUZZ_CONSTRAINTS: 
                if not isinstance(arg, dict): return "Error: Input must be dict"
                return self.sat.fuzz_constraints(state, arg)
            elif action == ActionType.DECODE_SOLUTION: return self.sat.decode_solution(state)
            elif action == ActionType.FINISH: return "Terminating"
            elif action == ActionType.ADD_CONSTRAINTS: return "Deprecated"
            elif action == ActionType.REFINE_MODEL: return self.sat.refine_model(state, arg)
            else: return f"Unknown Action {action}"
        except Exception as e: return f"Exec Error: {e}"

    def _parse_llm_output(self, raw: str) -> Dict[str, Any]:
        # Strip Markdown Code Blocks
        clean_raw = raw.strip()
        if clean_raw.startswith("```"):
            clean_raw = clean_raw.split("\n", 1)[1] # remove first line
            if clean_raw.endswith("```"):
                clean_raw = clean_raw[:-3].strip()
        
        try:
            data = json.loads(clean_raw)
        except Exception:
             # Try unescaping if it's a string repr of a dict (common weak LLM failure)
             try:
                 import ast
                 data = ast.literal_eval(clean_raw)
             except:
                 pass

        if "action" not in data: raise ValueError("No action")
        
        # --- ROBUSTNESS PATCH FOR LOCAL MODELS ---
        act = data["action"]
        
        # 1. Map DECOMPOSE -> DEFINE_VARIABLES
        if act == "DECOMPOSE":
            act = "DEFINE_VARIABLES"
            if isinstance(data.get("action_input"), dict):
                 inp = data["action_input"]
                 if "variables" in inp:
                     new_vars = []
                     for v in inp["variables"]:
                         if isinstance(v, dict): new_vars.append(v.get("name"))
                         elif isinstance(v, str): new_vars.append(v)
                     data["action_input"] = new_vars

        # 2. Map COMPILE -> SOLVE
        if act == "COMPILE": act = "SOLVE"

        if act == "CREATE_VARIABLES": act = "DEFINE_VARIABLES"
        if act == "DECLARE_VARIABLES": act = "DEFINE_VARIABLES"
        if act == "CREATE_CONSTRAINTS": act = "ADD_MODEL_CONSTRAINTS"
        if act == "ADD_CONSTRAINTS": act = "ADD_MODEL_CONSTRAINTS"
        
        data["action"] = act
        # -----------------------------------------

        try: data["action"] = ActionType(data["action"])
        except: raise ValueError(f"Invalid ActionType: {data.get('action')}")
        return data

    def _record(self, state, thought, act, obs):
        state.trajectory.append((thought, act, obs))

    def _construct_prompt(self, goal: str, state: AgentState) -> str:
        prompt = f"{self.system_prompt}\nGOAL: {goal}\nACTIVE_BACKEND: {state.active_ir_backend}\n"
        prompt += f"AVAILABLE: {self.registry.list_backends()}\n"
        prompt += f"SCHEMA: {self.sat.get_schema(state)}\n"
        
        # --- CONTEXT BOX (User Requested) ---
        prompt += "\n" + "="*40 + "\n"
        prompt += "[CONTEXT STATE]\n"
        
        # Variables
        vars_list = list(state.sat_variables.keys())
        prompt += f"REGISTERED VARIABLES ({len(vars_list)}):\n"
        if len(vars_list) > 100:
             prompt += f"{vars_list[:100]}... (+{len(vars_list)-100} more)\n"
        else:
             prompt += f"{vars_list}\n"

        # Constraints
        if state.model_constraints:
            prompt += f"\nREGISTERED CONSTRAINTS ({len(state.model_constraints)}):\n"
            for c in state.model_constraints:
                 # Telemetry Hint
                 cost = ""
                 if state.compile_report and c.id in state.compile_report.get("clauses_by_constraint_id", {}):
                     n_cls = state.compile_report["clauses_by_constraint_id"][c.id]
                     if n_cls > 100: cost = f" [HEAVY: {n_cls} clauses]"
                 prompt += f"- {c.id} | {c.kind} | {json.dumps(c.parameters)}{cost}\n"
        else:
            prompt += "(None)\n"
        
        if state.plan:
             prompt += "\nCURRENT PLAN:\n"
             prompt += json.dumps(state.plan, indent=2) + "\n"
        
        # Schema Reminder
        prompt += "\n" + "="*40 + "\n"
        prompt += "SCHEMA REMINDER:\n"
        try:
             backend = self.sat.registry.get(state.active_ir_backend)
             allowed = list(backend.allowed_kinds().keys())
             prompt += f"Active Backend: {state.active_ir_backend}\n"
             prompt += f"Allowed Constraint Kinds: {allowed}\n"
             prompt += f"Actions: {[a.value for a in ActionType]}\n"
        except: pass
        prompt += "="*40 + "\n"

        prompt += "\nHISTORY:\n"
        for i, (t, a, o) in enumerate(state.trajectory):
            prompt += f"{i}. T:{t} A:{a} O:{o}\n"
        return prompt
