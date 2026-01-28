import argparse
import sys
import os
import re

# Add project root to path to import engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mzn_to_fzn import compile_to_flatzinc, parse_flatzinc
from engine.booleanizer import Booleanizer

class DimacsFormatter:
    @staticmethod
    def write(clauses, num_vars, outfile):
        with open(outfile, 'w') as f:
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                # 0 is the terminator
                line = " ".join(map(str, clause)) + " 0\n"
                f.write(line)

class Translator:
    def __init__(self):
        self.booleanizer = Booleanizer()
        self.clauses = []

    def _get_lit(self, fzn_arg):
        """Resolves a FlatZinc argument to a literal."""
        # Argument could be "x", "true", "false"
        str_arg = str(fzn_arg).strip()
        if str_arg == "true": return None # Always true
        if str_arg == "false": return None # Always false
        
        # Check simple boolean check using Booleanizer
        # Assuming parse_flatzinc returned simple names
        try:
            return self.booleanizer.get_bool_literal(str_arg)
        except:
            return None # Handle ints later

    def translate_constraints(self, constraints):
        for c in constraints:
            ctype = c['type']
            args_str = c['args']
            
            # Simple parsing of comma-separated args
            # This is brittle for arrays [a,b,c] but works for basic bool_clause(a, b)
            args = [a.strip() for a in args_str.split(',')]
            
            if ctype == "bool_clause":
                # bool_clause(pos, neg)
                # This is a raw clause!  (x \/ y \/ ~z)
                # First arg is array of positive lits, second is array of negative lits
                # Parsing arrays from FZN string: "[x, y], [z]"
                
                # Rudimentary array parser
                m = re.match(r"\[(.*)\],\s*\[(.*)\]", args_str)
                if m:
                    pos_part = m.group(1).split(',') if m.group(1).strip() else []
                    neg_part = m.group(2).split(',') if m.group(2).strip() else []
                    
                    clause = []
                    for p in pos_part:
                        p = p.strip()
                        if not p: continue
                        l = self.booleanizer.get_bool_literal(p)
                        clause.append(l)
                    for n in neg_part:
                        n = n.strip()
                        if not n: continue
                        l = self.booleanizer.get_bool_literal(n)
                        clause.append(-l)
                    
                    self.clauses.append(clause)
            
            # TODO: Add int_lin_le, bool_or, etc.
            # Currently only supporting raw clauses output by Chuffed/MZN
            
    def process(self, mzn_file, output_cnf):
        print(f"1. Compiling {mzn_file}...")
        fzn_content = compile_to_flatzinc(mzn_file)
        
        print("2. Parsing FlatZinc...")
        vars_found, constrs_found = parse_flatzinc(fzn_content)
        
        print(f"3. Registering {len(vars_found)} variables...")
        for v in vars_found:
            if v['type'] == 'bool':
                self.booleanizer.register_bool(v['name'])
            # Integers would go here with domain parsing
            
        print(f"4. Translating {len(constrs_found)} constraints...")
        self.translate_constraints(constrs_found)
        
        print(f"5. Writing DIMACS to {output_cnf}")
        DimacsFormatter.write(self.clauses, self.booleanizer.next_literal_id - 1, output_cnf)
        print("Done.")

def main():
    parser = argparse.ArgumentParser(description="MiniZinc to DIMACS CNF Converter")
    parser.add_argument("mzn_file", help="Input .mzn file")
    args = parser.parse_args()
    
    target_cnf = args.mzn_file.replace('.mzn', '.cnf')
    
    translator = Translator()
    translator.process(args.mzn_file, target_cnf)

if __name__ == "__main__":
    main()
