# Implementation Plan

## Verification Status
**CONFIRMED**

## Observations
[
  "Grid is 8x8. T=20 timesteps (0 to 19). 4 robots. Movement: 4-neighbor or wait. Collision: no vertex collision, no edge swap. Goals enforced at t=T-1. Obstacles specified.",
  "MiniZinc backend. Using `ADD_MINIZINC_CODE` or `UPDATE_MODEL_FILE`."
]

## Variables
[
  "pos[r, t, x, y]: bool - true if robot r is at (x,y) at time t."
]

## Constraints
[
  "1. Initial positions: Each robot r must be at its start[r] position at t=0.",
  "2. Goal positions: Each robot r must be at its goal[r] position at t=T-1.",
  "3. Exactly one position: For each robot r and time t, `sum(bool2int(pos[r, t, x, y] for x,y))` must be 1.",
  "4. Movement constraints: For each robot r and time t < T-1, if pos[r, t, x, y] is true, then pos[r, t+1, x', y'] must be true for some (x', y') in the 4-neighbors of (x,y) or (x,y) itself.",
  "5. Obstacle avoidance: For any blocked cell (x_b, y_b), pos[r, t, x_b, y_b] must be false for all r, t.",
  "6. No vertex collision: For any time t and cell (x, y), at most one robot r can have pos[r, t, x, y] true. (i.e. `sum(bool2int(pos[r, t, x, y] for r)) <= 1`)",
  "7. No edge swap collision: For any time t < T-1 and distinct robots r1, r2, if r1 is at (x1,y1) and r2 is at (x2,y2) at time t, then it is not allowed for r1 to move to (x2,y2) and r2 to move to (x1,y1) simultaneously at time t+1."
]

## Strategy
Define constants and arrays for grid dimensions, robots, timesteps, start/goal positions, and blocked cells. Declare a 4D boolean array for robot positions `pos[robot_idx, time_idx, x_coord, y_coord]`. Implement all constraints using MiniZinc's `forall` loops and logical expressions, replacing `exactly_one` with `sum(bool2int(vars)) = 1`. I will use `UPDATE_MODEL_FILE` with `overwrite` mode to ensure the entire model is correct and to avoid issues with previous partial additions.

## Current Code
```json
[]
```

## Problems
[]

## Generated Code (Backend: minizinc)
```text
int: grid_w = 8;
int: grid_h = 8;
int: T = 20;
int: num_robots = 4;

set of int: Robots = 1..num_robots;
set of int: Time = 0..T-1;
set of int: Coords_X = 0..grid_w-1;
set of int: Coords_Y = 0..grid_h-1;

% Start and goal positions for robots (1-indexed for MiniZinc arrays)
array[Robots] of Coords_X: start_x_coords = [0, 7, 0, 7];
array[Robots] of Coords_Y: start_y_coords = [0, 0, 7, 7];
array[Robots] of Coords_X: goal_x_coords = [7, 0, 7, 0];
array[Robots] of Coords_Y: goal_y_coords = [7, 7, 0, 0];

% Blocked cells (1-indexed for MiniZinc arrays)
int: num_blocked = 8;
array[1..num_blocked] of Coords_X: blocked_x_coords = [3, 3, 4, 4, 2, 2, 5, 5];
array[1..num_blocked] of Coords_Y: blocked_y_coords = [3, 4, 3, 4, 3, 4, 3, 4];

% Decision variables
array[Robots, Time, Coords_X, Coords_Y] of var bool: pos;

% Constraints

% 1. Initial positions
constraint forall(r in Robots) (
    pos[r, 0, start_x_coords[r], start_y_coords[r]]
);

% 2. Goal positions
constraint forall(r in Robots) (
    pos[r, T-1, goal_x_coords[r], goal_y_coords[r]]
);

% 3. Exactly one position per robot per timestep
constraint forall(r in Robots, t in Time) (
    exactly_one([pos[r, t, x, y] | x in Coords_X, y in Coords_Y])
);

% 4. Movement constraints (4-neighbor + wait)
constraint forall(r in Robots, t in 0..T-2) (
    forall(x in Coords_X, y in Coords_Y) (
        pos[r, t, x, y] -> (
            pos[r, t+1, x, y] % Wait
            \/ (x > 0 /\ pos[r, t+1, x-1, y]) % Move left
            \/ (x < grid_w-1 /\ pos[r, t+1, x+1, y]) % Move right
            \/ (y > 0 /\ pos[r, t+1, x, y-1]) % Move down
            \/ (y < grid_h-1 /\ pos[r, t+1, x, y+1]) % Move up
        )
    )
);

% 5. Obstacle avoidance
constraint forall(r in Robots, t in Time, b in 1..num_blocked) (
    not pos[r, t, blocked_x_coords[b], blocked_y_coords[b]]
);

% 6. No vertex collision
constraint forall(t in Time, x in Coords_X, y in Coords_Y) (
    sum(r in Robots) (bool2int(pos[r, t, x, y])) <= 1
);

% 7. No edge swap collision
constraint forall(r1 in Robots, r2 in Robots where r1 < r2, t in 0..T-2) (
    forall(x1 in Coords_X, y1 in Coords_Y, x2 in Coords_X, y2 in Coords_Y where (x1 != x2 \/ y1 != y2)) (
        not (pos[r1, t, x1, y1] /\ pos[r2, t, x2, y2] /\
             pos[r1, t+1, x2, y2] /\ pos[r2, t+1, x1, y1])
    )
);

solve satisfy;

```
