import argparse
import subprocess
import re
import sys
import os

def compile_to_flatzinc(mzn_file):
    """Compiles a MiniZinc file to FlatZinc using the CLI."""
    try:
        # We output to stdout to capture it in memory, or we could use a temp file.
        # --compile is essentially -c. --output-fzn-to-stdout works on newer versions.
        # If not supported, we might need to output to a file. Let's try stdout first.
        # Some minizinc versions default to creating a .fzn file with the same name.
        
        # Let's be safe and write to a specific FZN file
        fzn_file = mzn_file.replace('.mzn', '.fzn')
        
        # Use absolute path to ensure it works even if PATH isn't updated in this shell
        minizinc_exe = "/Applications/MiniZincIDE.app/Contents/Resources/minizinc"
        if not os.path.exists(minizinc_exe):
            minizinc_exe = "minizinc" # Fallback
            
        # Must specify solver for flatzinc generation
        cmd = [minizinc_exe, "--compile", "--solver", "chuffed", mzn_file, "--output-to-file", fzn_file]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error compiling MiniZinc:\n{result.stderr}")
            sys.exit(1)
            
        if not os.path.exists(fzn_file):
             # Try checking if it wrote to stdout or default name
             print("Warning: explicit fzn file not found, checking output...")
             return result.stdout

        with open(fzn_file, 'r') as f:
            content = f.read()
        return content

    except FileNotFoundError:
        print("Error: 'minizinc' executable not found in PATH.")
        sys.exit(1)

def parse_flatzinc(fzn_content):
    """
    Parses FlatZinc content to extract variables and constraints.
    Ref: https://www.minizinc.org/doc-2.5.5/en/flatzinc_spec.html
    """
    variables = []
    constraints = []
    
    lines = fzn_content.splitlines()
    
    # Regex for variable declarations
    # var bool: x_INTRODUCED_0 :: var_is_introduced;
    # var 1..10: y;
    # array [1..5] of var int: z; (FlatZinc unrolls arrays usually, but keep eye out)
    
    var_pattern = re.compile(r"var\s+(bool|int|\d+\.\.\d+|float)\s*:\s*([a-zA-Z0-9_]+)(?:\s*::\s*.*)?;")
    
    # Regex for constraints
    # constraint bool_or(x, y, z);
    # constraint int_le(a, b);
    constraint_pattern = re.compile(r"constraint\s+([a-zA-Z0-9_]+)\((.*)\);")

    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
            
        # Check Var
        v_match = var_pattern.match(line)
        if v_match:
            v_type = v_match.group(1)
            v_name = v_match.group(2)
            variables.append({"name": v_name, "type": v_type, "raw": line})
            continue
            
        # Check Constraint
        c_match = constraint_pattern.match(line)
        if c_match:
            c_type = c_match.group(1)
            c_args = c_match.group(2)
            # Add implicit grouping by index/ID
            # This is critical for the Debug Harness to attribute failures
            c_id = f"fzn_c_{len(constraints)}"
            constraints.append({
                "type": c_type, 
                "args": c_args, 
                "raw": line,
                "group": c_id # Unique ID for harness
            })
            continue

    return variables, constraints

def main():
    parser = argparse.ArgumentParser(description="Convert mzn to fzn and parse it.")
    parser.add_argument("mzn_file", help="Path to the .mzn file")
    args = parser.parse_args()
    
    print(f"--- Compiling {args.mzn_file} ---")
    fzn_content = compile_to_flatzinc(args.mzn_file)
    
    print(f"--- Parsing FlatZinc ---")
    vars_found, constrs_found = parse_flatzinc(fzn_content)
    
    print(f"Found {len(vars_found)} variables.")
    print(f"Found {len(constrs_found)} constraints.")
    
    print("\n--- Variables ---")
    for v in vars_found:
        print(f"  {v['name']} ({v['type']})")

    print("\n--- Constraints ---")
    for c in constrs_found:
        print(f"  {c['type']}: {c['args']}")

if __name__ == "__main__":
    main()
