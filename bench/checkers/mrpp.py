from typing import Dict, List, Tuple, Any

def check(solution: Dict[str, Any], instance: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates the MRPP solution against the instance.
    
    Args:
        solution: The solution dictionary returned by the agent.
                  Expected structure: {"paths": {robot_name: [[x,y], ...]}}
        instance: The instance dictionary.
                  Expected structure: {"data": {"grid_w": int, "grid_h": int, "T": int,
                                                "robots": [{"name": str, "start": [x,y], "goal": [x,y]}],
                                                "blocked": [[x,y], ...]}}

    Returns:
        (True, []) if valid.
        (False, [errors]) if invalid.
    """
    errors = []
    
    if "paths" not in solution:
        return False, ["Solution missing 'paths' key."]
    
    paths = solution["paths"]
    data = instance["data"]
    
    T = data["T"]
    grid_w = data["grid_w"]
    grid_h = data["grid_h"]
    blocked = set(tuple(b) for b in data.get("blocked", []))
    
    robots = {r["name"]: r for r in data["robots"]}
    
    # 1. Validate all robots have paths
    for r_name in robots:
        if r_name not in paths:
            errors.append(f"Missing path for robot {r_name}.")
    
    if len(errors) > 0:
        return False, errors

    # Pre-process paths for collision checks
    # path_steps[t][r_name] = (x, y)
    path_steps = []
    for t in range(T + 1):
        path_steps.append({})

    for r_name, path in paths.items():
        if r_name not in robots:
            errors.append(f"Unknown robot {r_name} in solution.")
            continue
            
        # 2. Check path length
        if len(path) != T + 1:
            errors.append(f"Path for {r_name} has length {len(path)}, expected {T + 1}.")
            continue
            
        robot_spec = robots[r_name]
        start_pos = tuple(robot_spec["start"])
        goal_pos = tuple(robot_spec["goal"])
        
        # 3. Check start and goal
        if tuple(path[0]) != start_pos:
            errors.append(f"Robot {r_name} starts at {path[0]}, expected {start_pos}.")
        if tuple(path[-1]) != goal_pos:
            errors.append(f"Robot {r_name} ends at {path[-1]}, expected {goal_pos}.")
            
        for t, pos in enumerate(path):
            pos_tuple = tuple(pos)
            x, y = pos_tuple
            
            # 4. Check bounds
            if not (0 <= x < grid_w and 0 <= y < grid_h):
                errors.append(f"Robot {r_name} at step {t} is out of bounds: {pos}.")
            
            # 5. Check blocked cells
            if pos_tuple in blocked:
                errors.append(f"Robot {r_name} at step {t} hits blocked cell {pos}.")
                
            path_steps[t][r_name] = pos_tuple
            
            # 6. Check moves (continuity)
            if t > 0:
                prev_pos = tuple(path[t-1])
                dx = abs(x - prev_pos[0])
                dy = abs(y - prev_pos[1])
                if not ((dx == 0 and dy == 0) or (dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
                    errors.append(f"Robot {r_name} makes invalid move from {prev_pos} to {pos} at step {t}.")

    if len(errors) > 0:
        return False, errors

    # Collision Checks
    for t in range(T + 1):
        # 7. Vertex collisions
        positions = {}
        for r_name, pos in path_steps[t].items():
            if pos in positions:
                other = positions[pos]
                errors.append(f"Vertex collision at step {t} between {r_name} and {other} at {pos}.")
            positions[pos] = r_name
            
        # 8. Edge collisions (swaps)
        if t > 0:
            for r_name, curr_pos in path_steps[t].items():
                prev_pos = path_steps[t-1][r_name]
                # Check for any other robot that moved from curr_pos to prev_pos
                for other_name, other_curr_pos in path_steps[t].items():
                    if r_name == other_name:
                        continue
                    other_prev_pos = path_steps[t-1][other_name]
                    
                    if curr_pos == other_prev_pos and prev_pos == other_curr_pos:
                         errors.append(f"Edge collision (swap) between {r_name} and {other_name} at steps {t-1}->{t}.")

    valid = len(errors) == 0
    return valid, errors
