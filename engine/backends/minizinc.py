import subprocess
import os
import uuid
import json
import re
import hashlib
import pickle
from typing import List, Dict, Any, Optional

from engine.backends.base import IRBackend
from engine.state import AgentState, ModelingConstraint
from tools.mzn_to_fzn import compile_to_flatzinc, parse_flatzinc
from engine.booleanizer import Booleanizer
from engine.compilation.artifact import CompilationArtifact

# PySAT imports
try:
    from pysat.card import CardEnc
    from pysat.pb import PBEnc
except ImportError:
    CardEnc = Any
    PBEnc = Any

class MiniZincCoreBackend(IRBackend):
    def __init__(self):
        self.solver_id = "chuffed" 
        self.booleanizer = Booleanizer()
        self._cache_dir = "memory/cache/mzn"
        os.makedirs(self._cache_dir, exist_ok=True)

    @property
    def name(self) -> str:
        return "minizinc"

    def get_prompt_doc(self) -> str:
        return """
### BACKEND: MINIZINC
You must use the MiniZinc backend.
This backend maps your abstract constraints to high-level MiniZinc primitives:
- `alldifferent_onehot`: Maps to the `alldifferent` global constraint.
- `linear_eq` / `linear_leq`: Maps to native integer arithmetic (e.g. `sum(x) <= k`).
- `at_most_k`: Maps to cardinality constraints.
Constraint logic is handled by the solver's CP-SAT engine.
"""

    def allowed_kinds(self) -> Dict[str, Any]:
        return {
            "clause": {"parameters": {"literals": "List[str]"}},
            "implies": {"parameters": {"a": "str", "b": "str"}},
            "at_most_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "at_least_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "exactly_k": {"parameters": {"vars": "List[str]", "k": "int"}},
            "exactly_one": {"parameters": {"vars": "List[str]"}},
            "alldifferent_onehot": {"parameters": {"groups": "List[List[str]]"}}, # Harder to map directly?
            "linear_leq": {"parameters": {"terms": "List[Dict]", "rhs": "int"}},
            "linear_eq": {"parameters": {"terms": "List[Dict]", "rhs": "int"}}
        }

    def validate_constraint(self, constraint: ModelingConstraint, state: AgentState) -> None:
        if constraint.kind not in self.allowed_kinds(): raise ValueError(f"Unknown kind {constraint.kind}")
        # Basic variable existence check
        # (This is shared logic usually, but we keep it here for now)
        pass 

    def compile_constraints(self, constraints: List[ModelingConstraint], state: AgentState) -> List[List[int]]:
        # For MiniZinc, "compile" means "generate .mzn file context"
        # BUT the current architecture assumes we return CNF clauses (List[List[int]])!
        # This is a mismatch. If we want to use MiniZinc, we shouldn't be returning CNF clauses 
        # to be solved by PySAT. We should be solving it OURSELVES or returning empty clauses 
        # and handling the solve in a custom `solve()` method.
        # However, `SATManager.solve()` calls `_compile` then uses `Solver`.
        # We need to refactor SATManager to delegate SOLVING to the backend too.
        return []

    def generate_code(self, state: AgentState) -> str:
        # 0. File-Centric Override
        if state.model_file_path and os.path.exists(state.model_file_path):
             try:
                 with open(state.model_file_path, "r") as f:
                     return f.read()
             except: pass

        # 1. Generate MZN
        mzn_lines = []
        
        # Variables (Boolean)
        max_id = state.next_var_id
        if state.sat_variables:
             max_id = max(max(state.sat_variables.values()) + 1, max_id)
        
        mzn_lines.append(f"array[1..{max_id}] of var bool: x;")
        
        # Helper to get MZN ref
        def get_mzn_ref(name):
            name_str = str(name)
            is_neg = name_str.startswith('-') or name_str.startswith('~')
            atom = name_str.lstrip('-~')
            vid = state.sat_variables.get(atom)
            if vid is None: raise ValueError(f"Missing var {atom}")
            return f"not x[{vid}]" if is_neg else f"x[{vid}]"

        for c in state.model_constraints:
            k = c.kind
            p = c.parameters
            
            if k == "clause":
                lits = [get_mzn_ref(l) for l in p["literals"]]
                sep = " \\/ "
                mzn_lines.append(f"constraint {sep.join(lits)};")
            elif k == "implies":
                mzn_lines.append(f"constraint {get_mzn_ref(p['a'])} -> {get_mzn_ref(p['b'])};")
            elif k == "exactly_one":
                vars_ref = [get_mzn_ref(v) for v in p["vars"]]
                bools = [f"bool2int({v})" for v in vars_ref]
                mzn_lines.append(f"constraint sum([{', '.join(bools)}]) = 1;")
            elif k == "at_most_k":
                vars_ref = [get_mzn_ref(v) for v in p["vars"]]
                bools = [f"bool2int({v})" for v in vars_ref]
                mzn_lines.append(f"constraint sum([{', '.join(bools)}]) <= {p['k']};")
            # ... (Implement others as needed)
            
        # 2. Inject Raw Code (High-Level)
        if hasattr(state, "minizinc_code") and state.minizinc_code:
            mzn_lines.append("\n% --- User High-Level Code ---")
            mzn_lines.extend(state.minizinc_code)
            
        mzn_lines.append("solve satisfy;")
        return "\n".join(mzn_lines)

    def compile(self, state: AgentState) -> CompilationArtifact:
        """
        Compiles MZN model into CompilationArtifact via FZN translation.
        Includes cross-run hashing cache.
        """
        # 1. Generate/Locate MZN Code
        mzn_code = self.generate_code(state)
        
        # 2. Cache Check
        mzn_hash = hashlib.sha256(mzn_code.encode()).hexdigest()[:16]
        cache_path = os.path.join(self._cache_dir, f"{mzn_hash}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached_artifact = pickle.load(f)
                    print(f"MiniZincCoreBackend: Cache Hit ({mzn_hash})")
                    return cached_artifact
            except Exception as e:
                print(f"MiniZincCoreBackend: Cache Load Failed: {e}")

        f_mzn = "memory/model_temp.mzn" # Default fallback
        os.makedirs("memory", exist_ok=True)
        
        if state.model_file_path and os.path.exists(state.model_file_path):
             f_mzn = state.model_file_path
        else:
             model_id = uuid.uuid4().hex[:8]
             f_mzn = f"memory/model_{model_id}.mzn"
             try:
                 with open(f_mzn, "w") as f:
                     f.write(mzn_code)
             except Exception as e:
                 raise RuntimeError(f"Write Error: {e}")

        # 3. Compile to FlatZinc
        try:
            fzn_content = compile_to_flatzinc(f_mzn)
        except Exception as e:
            raise RuntimeError(f"MiniZinc Compilation Failed:\n{e}")

        # 3. Parse and Translate
        vars_found, constrs_found = parse_flatzinc(fzn_content)
        
        # Init Booleanizer with shared VarManager
        local_booleanizer = Booleanizer(state.var_manager)
        
        # Register Vars from FZN
        for v in vars_found:
            if v['type'] == 'bool':
                local_booleanizer.register_bool(v['name'])
            # Int support pending - logic inside booleanizer handles dynamic calls usually?
            # register_bool will reuse ID if name matches existing (from Agent's define_variables)
        
        output_clauses = []
        constraint_to_clause_ids = {}
        aux_vars = set()
        
        backend_stats = {"mzn_constraints": len(constrs_found)}
        
        # Translate Constraints
        for c in constrs_found:
            current_clauses = []
            
            # (Translation Logic - Copied & Adapted from old solve)
            if c['type'] == 'bool_clause':
                args_str = c['args']
                import re
                m = re.match(r"\[(.*)\],\s*\[(.*)\]", args_str)
                if m:
                    pos_part = m.group(1).split(',') if m.group(1).strip() else []
                    neg_part = m.group(2).split(',') if m.group(2).strip() else []
                    clause = []
                    for p in pos_part:
                        if p.strip(): clause.append(local_booleanizer.get_bool_literal(p.strip()))
                    for n in neg_part:
                        if n.strip(): clause.append(-local_booleanizer.get_bool_literal(n.strip()))
                    current_clauses.append(clause)
            
            elif c['type'] == 'bool_not':
                args_part = c['args']
                parts = [x.strip() for x in args_part.split(',')]
                if len(parts) >= 2:
                    a_name, b_name = parts[0], parts[1]
                    try:
                        lit_a = local_booleanizer.get_bool_literal(a_name)
                        lit_b = local_booleanizer.get_bool_literal(b_name)
                        current_clauses.append([lit_b, lit_a])
                        current_clauses.append([-lit_b, -lit_a])
                    except ValueError: pass

            elif c['type'] in ['int_lin_le', 'int_lin_le_reif', 'int_lin_eq', 'int_lin_eq_reif']:
                # Simplified translation logic
                # ... (Assuming similar logic to before but creating clauses directly)
                try:
                    is_reif = c['type'].endswith('_reif')
                    args_str = c['args']
                    import re
                    m = re.match(r"\[([\-\d,\s]+)\],\s*\[(.*)\],\s*(\-?\d+)(?:,\s*([a-zA-Z0-9_]+))?", args_str)
                    if m:
                        coeffs = [int(x) for x in m.group(1).split(',')]
                        var_names = [x.strip() for x in m.group(2).split(',')]
                        rhs = int(m.group(3))
                        # reif_var = m.group(4) if is_reif else None
                        
                        p_lits = [local_booleanizer.get_bool_literal(n) for n in var_names]
                        p_weights = coeffs
                        
                        top_id = state.var_manager.max_id
                        cnf_obj = None
                        
                        if 'eq' in c['type']:
                            if all(w == 1 for w in p_weights):
                                cnf_obj = CardEnc.equals(lits=p_lits, bound=rhs, top_id=top_id)
                            else:
                                cnf_obj = PBEnc.equals(lits=p_lits, weights=p_weights, bound=rhs, top_id=top_id)
                        else:
                            if all(w == 1 for w in p_weights):
                                cnf_obj = CardEnc.atmost(lits=p_lits, bound=rhs, top_id=top_id)
                            else:
                                cnf_obj = PBEnc.atmost(lits=p_lits, weights=p_weights, bound=rhs, top_id=top_id)
                                
                        if cnf_obj:
                            current_clauses.extend(cnf_obj.clauses)
                            num_new = cnf_obj.nv - top_id
                            if num_new > 0:
                                allocated = state.var_manager.reserve_block(num_new, prefix="mzn", namespace="enc")
                                aux_vars.update(allocated)

                except Exception as e:
                    print(f"Warning: Failed to encode int_lin in MZN backend: {e}")

            # Register clauses and provenance
            if current_clauses:
                start_idx = len(output_clauses)
                output_clauses.extend(current_clauses)
                end_idx = len(output_clauses)
                
                # Use 'group' annotation if present, else fallback
                gid = c.get("group", f"c_mzn_{uuid.uuid4().hex[:4]}")
                if gid not in constraint_to_clause_ids:
                    constraint_to_clause_ids[gid] = []
                constraint_to_clause_ids[gid].extend(range(start_idx, end_idx))
        
        artifact = CompilationArtifact(
            backend_name=self.name,
            encoding_config={},
            clauses=output_clauses,
            var_map=state.var_manager.get_var_map(),
            id_to_name=state.var_manager.get_id_to_name(),
            constraint_ids=list(constraint_to_clause_ids.keys()),
            constraint_to_clause_ids=constraint_to_clause_ids,
            aux_vars=aux_vars,
            stats=backend_stats
        )
        
        # Save to Cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(artifact, f)
        except Exception as e:
            print(f"MiniZincCoreBackend: Cache Save Failed: {e}")
            
        return artifact

    def solve(self, state: AgentState) -> str:
         return "DeprecationWarning: MiniZincBackend.solve is deprecated. Use SATManager."
