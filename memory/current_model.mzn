int: grid_w = 6;
int: grid_h = 6;
int: T = 8;
int: num_robots = 3;
set of int: R = 1..num_robots;
set of int: T_RANGE = 1..T;
set of int: X_RANGE = 1..grid_w;
set of int: Y_RANGE = 1..grid_h;

array[R, T_RANGE, X_RANGE, Y_RANGE] of var bool: pos;

% Blocked cells (1-indexed)
array[1..8, 1..2] of int: blocked_cells = [|2,2 | 2,3 | 2,4 | 5,2 | 5,3 | 5,4 | 3,4 | 4,3|];

% Robot start/goal positions (1-indexed)
array[R] of int: start_x = [1, 1, 3];
array[R] of int: start_y = [1, 6, 1];
array[R] of int: goal_x = [6, 6, 3];
array[R] of int: goal_y = [6, 1, 6];

% C1: Each robot must occupy exactly one cell at each timestep.
constraint forall (r in R, t in T_RANGE) (
    sum([bool2int(pos[r, t, x, y]) | x in X_RANGE, y in Y_RANGE]) == 1
  );

% C2: Robots must start at their specified start positions.
constraint forall (r in R) (
    pos[r, 1, start_x[r], start_y[r]]
 );

% C3: Robots must end at their specified goal positions.
constraint forall (r in R) (
    pos[r, T, goal_x[r], goal_y[r]]
 );

% C4: Robots cannot be in a blocked cell at any time.
constraint forall (r in R, t in T_RANGE, b in 1..8) (
    not pos[r, t, blocked_cells[b,1], blocked_cells[b,2]]
 );

% C5: Movement constraints: position at t+1 must be adjacent or same as at t.
constraint forall (r in R, t in 1..T-1, x_curr in X_RANGE, y_curr in Y_RANGE) (
    if pos[r, t, x_curr, y_curr] then (
        sum([bool2int(pos[r, t+1, x_next, y_next])
             | x_next in X_RANGE, y_next in Y_RANGE
             where abs(x_curr - x_next) + abs(y_curr - y_next) <= 1]) == 1
    ) endif
 );

% C6: Vertex collision: No two robots can occupy the same cell at the same timestep.
constraint forall (t in T_RANGE, x in X_RANGE, y in Y_RANGE) (
    sum(r in R)(bool2int(pos[r, t, x, y])) <= 1
 );

% C7: Edge collision: No two robots can swap positions between two adjacent timesteps.
constraint forall (r1 in R, r2 in R where r1 < r2, t in 1..T-1, x1 in X_RANGE, y1 in Y_RANGE, x2 in X_RANGE, y2 in Y_RANGE) (
    if (x1 != x2 \/ y1 != y2) then
        not (pos[r1, t, x1, y1] /\ pos[r2, t, x2, y2] /\ pos[r1, t+1, x2, y2] /\ pos[r2, t+1, x1, y1])
    endif
 );

solve satisfy;