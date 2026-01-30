int: grid_w = 6;
int: grid_h = 6;
int: T = 8;
int: num_robots = 3;

array[0..num_robots-1] of int: start_x = [0, 0, 2];
array[0..num_robots-1] of int: start_y = [0, 5, 0];
array[0..num_robots-1] of int: goal_x = [5, 5, 2];
array[0..num_robots-1] of int: goal_y = [5, 0, 5];

set of int: ROBOTS = 0..num_robots-1;
set of int: TIME = 0..T-1;
set of int: GRID_X = 0..grid_w-1;
set of int: GRID_Y = 0..grid_h-1;

int: num_blocked = 8;
array[0..num_blocked-1, 0..1] of int: blocked_cells = [|
    1, 1 | 1, 2 | 1, 3 |
    4, 1 | 4, 2 | 4, 3 |
    2, 3 | 3, 2
|];

% Variables
array[ROBOTS, TIME, GRID_X, GRID_Y] of var bool: pos;
array[ROBOTS, TIME] of var GRID_X: robot_x;
array[ROBOTS, TIME] of var GRID_Y: robot_y;

% Constraints

% 1. Initial Position
constraint forall(r in ROBOTS) (
    pos[r, 0, start_x[r], start_y[r]]
);

% 2. Goal Position
constraint forall(r in ROBOTS) (
    pos[r, T-1, goal_x[r], goal_y[r]]
);

% 3. Exactly One Position & 4. Link pos to robot_x/robot_y
constraint forall(r in ROBOTS, t in TIME) (
    sum(x in GRID_X, y in GRID_Y) (int(pos[r, t, x, y])) = 1
    /\
    sum(x in GRID_X, y in GRID_Y) (x * int(pos[r, t, x, y])) = robot_x[r, t]
    /\
    sum(x in GRID_X, y in GRID_Y) (y * int(pos[r, t, x, y])) = robot_y[r, t]
);

% 5. Movement
constraint forall(r in ROBOTS, t in 0..T-2) (
    forall(x in GRID_X, y in GRID_Y where pos[r, t, x, y]) (
        exists(nx in GRID_X, ny in GRID_Y) (
            pos[r, t+1, nx, ny]
            /\
            abs(nx - x) + abs(ny - y) <= 1
        )
    )
);

% 6. Blocked Cells
constraint forall(r in ROBOTS, t in TIME, b in 0..num_blocked-1) (
    not pos[r, t, blocked_cells[b, 0], blocked_cells[b, 1]]
);

% 7. Vertex Collision
constraint forall(t in TIME, x in GRID_X, y in GRID_Y) (
    sum(r in ROBOTS) (int(pos[r, t, x, y])) <= 1
);

% 8. Edge Collision
constraint forall(t in 0..T-2, r1 in ROBOTS, r2 in ROBOTS where r1 < r2) (
    forall(x1 in GRID_X, y1 in GRID_Y, x2 in GRID_X, y2 in GRID_Y where x1 != x2 \/ y1 != y2) (
        not (
            pos[r1, t, x1, y1] /\ pos[r1, t+1, x2, y2] /\
            pos[r2, t, x2, y2] /\ pos[r2, t+1, x1, y1]
        )
    )
);

solve satisfy;

output
[
  "{\n" ++
  intercalate(",\n", [
    "\"R" ++ show(r+1) ++ "\": " ++ show([ [robot_x[r,t], robot_y[r,t]] | t in TIME ])
    | r in ROBOTS
  ]) ++
  "\n}\n"
];
