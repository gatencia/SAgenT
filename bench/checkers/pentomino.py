from typing import Dict, Any, List, Set, Tuple

# Standard 12 Pentomino definitions (as sets of relative (x,y) coords)
# We handle rotations/flips in the checker by attempting to match the shape.
PENTOMINOES = {
    "F": {(0, 1), (1, 1), (1, 0), (1, 2), (2, 2)},
    "I": {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)},
    "L": {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)},
    "P": {(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)}, 
    "N": {(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)},
    "T": {(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)},
    "U": {(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
    "V": {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
    "W": {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)},
    "X": {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
    "Y": {(0, 1), (1, 1), (2, 1), (3, 1), (1, 0)},
    "Z": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)},
}

def normalize_shape(coords: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Shift to (0,0)"""
    if not coords: return set()
    min_x = min(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    return {(x - min_x, y - min_y) for x, y in coords}

def get_variants(shape: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    """Generate all 8 symmetries."""
    variants = []
    curr = shape
    for _ in range(2): # Flip
        for _ in range(4): # Rotate
            variants.append(normalize_shape(curr))
            curr = {(y, -x) for x, y in curr}
        curr = {(x, -y) for x, y in curr}
    return variants

# Precompute variants
SHAPE_VARIANTS = {k: get_variants(v) for k, v in PENTOMINOES.items()}

def check(instance: Dict[str, Any], solution: Dict[str, bool]) -> str:
    # 1. Parse Grid
    # We infer grid assignment from variables.
    # LLM might use "cell_x_y_P" or "P_x_y_..."
    # We try to find True variables that look like assignments
    
    w, h = instance["data"]["grid_w"], instance["data"]["grid_h"]
    grid = {} # (x,y) -> piece_name
    
    # Heuristic parsing
    # Look for variables like "P_x_y" or "cell_x_y_P"
    for var, val in solution.items():
        if not val: continue
        parts = var.split('_')
        # Try to match patterns
        # Case 1: cell_x_y_P
        # Case 2: P_x_y
        
        # We need identifying mapping
        x, y, p = -1, -1, None
        
        # Simple scan for known pieces
        found_p = None
        for piece in PENTOMINOES:
            if piece in parts:
                found_p = piece
                break
        
        if not found_p: continue
        
        # Extract coords
        coords = []
        for part in parts:
            if part.isdigit(): coords.append(int(part))
            
        if len(coords) >= 2:
            x, y = coords[0], coords[1] # Assume order x, y
            # Check bounds
            if 0 <= x < w and 0 <= y < h:
                if (x,y) in grid and grid[(x,y)] != found_p:
                     return "FAIL: Overlapping pieces at ({}, {})".format(x,y)
                grid[(x,y)] = found_p

    # 2. Verify Exact Cover
    used_pieces = set(grid.values())
    all_pieces = set(PENTOMINOES.keys())
    
    missing = all_pieces - used_pieces
    if missing: return f"FAIL: Missing pieces: {missing}"
    
    extra = used_pieces - all_pieces
    if extra: return f"FAIL: Unknown pieces: {extra}" # Unlikely

    # 3. Verify Shapes
    for p in all_pieces:
        cells = {c for c, v in grid.items() if v == p}
        if len(cells) != 5: return f"FAIL: Piece {p} has {len(cells)} cells, expected 5."
        norm = normalize_shape(cells)
        if norm not in SHAPE_VARIANTS[p]:
            return f"FAIL: Piece {p} shape mismatch. Parsed: {norm}"

    # 4. Verify Connectivity (The "Fence")
    # All occupied cells must form ONE connected component
    occupied = set(grid.keys())
    if not occupied: return "FAIL: Empty grid"
    
    start = next(iter(occupied))
    queue = [start]
    seen = {start}
    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            n = (cx+dx, cy+dy)
            if n in occupied and n not in seen:
                seen.add(n)
                queue.append(n)
    
    if len(seen) != len(occupied):
        return f"FAIL: Disconnected components. Found size {len(seen)} vs total {len(occupied)}."

    # 5. Verify Loop / Enclosed Area
    # A fence must enclose a region.
    # We do a flood fill from the boundary of the grid (assuming grid is larger than fence).
    # If any empty cell is NOT reachable from the boundary, it is "inside".
    
    # Note: 20x20 grid is large enough for a 60-cell fence. 
    # Boundary (0,0) is likely outside.
    
    outside = set()
    q = []
    
    # Add border cells
    for i in range(w):
        if (i, 0) not in occupied: q.append((i,0)); outside.add((i,0))
        if (i, h-1) not in occupied: q.append((i,h-1)); outside.add((i,h-1))
    for j in range(h):
        if (0, j) not in occupied: q.append((0,j)); outside.add((0,j))
        if (w-1, j) not in occupied: q.append((w-1,j)); outside.add((w-1,j))
        
    while q:
        cx, cy = q.pop(0)
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w and 0 <= ny < h:
                if (nx, ny) not in occupied and (nx, ny) not in outside:
                    outside.add((nx, ny))
                    q.append((nx, ny))
                    
    # Calculate Enclosed Area
    total_cells = w * h
    occupied_count = len(occupied) # Should be 60
    outside_count = len(outside)
    inside_area = total_cells - occupied_count - outside_count
    
    target = instance["data"].get("min_area", 0)
    
    if inside_area < target:
        return f"FAIL: Area too small. Enclosed {inside_area}, needed {target}."
        
    if inside_area == 0:
        return "FAIL: No enclosed area (Not a loop?)."

    return f"PASS: Valid PentoFence. Area={inside_area}"
