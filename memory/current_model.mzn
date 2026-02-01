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
int: num_blocked_cells = 8;
array[1..num_blocked_cells] of Coords_X: blocked_x = [3, 3, 4, 4, 2, 2, 5, 5];
array[1..num_blocked_cells] of Coords_Y: blocked_y = [3, 4, 3, 4, 3, 4, 3, 4];

% Variable: pos[r, t, x, y] is true if robot r is at (x,y) at time t
array[Robots, Time, Coords_X, Coords_Y] of var bool: pos;

% 1. Initial positions (t=0)
constraint forall(r in Robots) (
    pos[r, 0, start_x_coords[r], start_y_coords[r]]
);
constraint forall(r in Robots, x in Coords_X, y in Coords_Y where x != start_x_coords[r] \/ y != start_y_coords[r]) (
    not pos[r, 0, x, y]
);

% 2. Goal positions (t=T-1)
constraint forall(r in Robots) (
    pos[r, T-1, goal_x_coords[r], goal_y_coords[r]]
);
constraint forall(r in Robots, x in Coords_X, y in Coords_Y where x != goal_x_coords[r] \/ y != goal_y_coords[r]) (
    not pos[r, T-1, x, y]
);

% 3. Exactly one position per robot per timestep
constraint forall(r in Robots, t in Time) (
    sum(x in Coords_X, y in Coords_Y) (bool2int(pos[r, t, x, y])) == 1
);

% 4. Movement constraints (4-neighbor + wait)
constraint forall(r in Robots, t in 0..T-2) (
    forall(x in Coords_X, y in Coords_Y) (
        pos[r, t, x, y] -> (
            exists(nx in Coords_X, ny in Coords_Y) (
                abs(x - nx) + abs(y - ny) <= 1 /\ 
                pos[r, t+1, nx, ny]
            )
        )
    )
);

% 5. Obstacle avoidance
constraint forall(r in Robots, t in Time, b_idx in 1..num_blocked_cells) (
    not pos[r, t, blocked_x[b_idx], blocked_y[b_idx]]
);

% 6. No vertex collision
constraint forall(t in Time, x in Coords_X, y in Coords_Y) (
    sum(r in Robots) (bool2int(pos[r, t, x, y])) <= 1
);

% 7. No edge swap collision
constraint forall(t in 0..T-2, r1 in Robots, r2 in Robots where r1 < r2) (
    forall(x1 in Coords_X, y1 in Coords_Y, x2 in Coords_X, y2 in Coords_Y where (x1 != x2 \/ y1 != y2)) (
        (pos[r1, t, x1, y1] /\ pos[r2, t, x2, y2]) -> (
            not (pos[r1, t+1, x2, y2] /\ pos[r2, t+1, x1, y1])
        )
    )
);

solve satisfy;
