array[1..2] of var bool: x;
constraint x[1];

% --- User High-Level Code ---
% Constants
int: grid_w = 6;
int: grid_h = 6;
int: T = 8;
int: num_robots = 3;

set of int: ROBOTS = 0..num_robots-1;
set of int: TIMES = 0..T;
set of int: COORDS_X = 0..grid_w-1;
set of int: COORDS_Y = 0..grid_h-1;

% Blocked cells
array[1..8, 1..2] of int: blocked_cells_data = array2d(1..8, 1..2, [|1, 1, |1, 2, |1, 3, |4, 1, |4, 2, |4, 3, |2, 3, |3, 2|]);

% Robot start and goal positions
array[ROBOTS] of int: start_x = [0, 0, 2];
array[ROBOTS] of int: start_y = [0, 5, 0];
array[ROBOTS] of int: goal_x = [5, 5, 2];
array[ROBOTS] of int: goal_y = [5, 0, 5];

% Main decision variables
array[ROBOTS, TIMES, COORDS_X, COORDS_Y] of var bool: pos;

% Constants
int: grid_w = 6;
int: grid_h = 6;
int: T = 8;
int: num_robots = 3;

set of int: ROBOTS = 0..num_robots-1;
set of int: TIMES = 0..T;
set of int: COORDS_X = 0..grid_w-1;
set of int: COORDS_Y = 0..grid_h-1;

% Blocked cells
array[1..8, 1..2] of int: blocked_cells_data = array2d(1..8, 1..2, [|1, 1, |1, 2, |1, 3, |4, 1, |4, 2, |4, 3, |2, 3, |3, 2|]);

% Robot start and goal positions
array[ROBOTS] of int: start_x = [0, 0, 2];
array[ROBOTS] of int: start_y = [0, 5, 0];
array[ROBOTS] of int: goal_x = [5, 5, 2];
array[ROBOTS] of int: goal_y = [5, 0, 5];

% Main decision variables
array[ROBOTS, TIMES, COORDS_X, COORDS_Y] of var bool: pos;

% Constraints:
% 1. Each robot must be at exactly one cell at each timestep.
constraint forall(r in ROBOTS, t in TIMES) (
  sum(x in COORDS_X, y in COORDS_Y) (bool2int(pos[r, t, x, y])) == 1
);

% 2. Robots must start at their designated start positions.
constraint forall(r in ROBOTS) (
  pos[r, 0, start_x[r], start_y[r]]
);

% 3. Robots must end at their designated goal positions.
constraint forall(r in ROBOTS) (
  pos[r, T, goal_x[r], goal_y[r]]
);

% 4. Movement constraints: A robot at (x,y) at time 't' must move to an adjacent cell or stay at (x,y) at time 't+1'.
constraint forall(r in ROBOTS, t in 0..T-1) (
  forall(x in COORDS_X, y in COORDS_Y) (
    pos[r, t, x, y] -> 
      (sum(dx in -1..1, dy in -1..1 where abs(dx) + abs(dy) <= 1 /\
         x+dx >= 0 /\ x+dx < grid_w /\
         y+dy >= 0 /\ y+dy < grid_h) (
        bool2int(pos[r, t+1, x+dx, y+dy])
      ) = 1)
  )
);

% 5. Blocked cells must never be occupied by any robot.
constraint forall(r in ROBOTS, t in TIMES, b in 1..8) (
  not pos[r, t, blocked_cells_data[b, 1], blocked_cells_data[b, 2]]
);

% 6. Collision avoidance: No two robots can occupy the same cell at the same time.
constraint forall(t in TIMES, x in COORDS_X, y in COORDS_Y) (
  sum(r in ROBOTS) (bool2int(pos[r, t, x, y])) <= 1
);

% 7. Head-on collision avoidance: Two robots cannot swap adjacent cells in one timestep.
constraint forall(t in 0..T-1, r1 in ROBOTS, r2 in ROBOTS where r1 < r2) (
  forall(x1 in COORDS_X, y1 in COORDS_Y, x2 in COORDS_X, y2 in COORDS_Y where (abs(x1-x2) + abs(y1-y2) == 1)) (
    not (pos[r1, t, x1, y1] /\
         pos[r2, t, x2, y2] /\
         pos[r1, t+1, x2, y2] /\
         pos[r2, t+1, x1, y1])
  )
);

solve satisfy;