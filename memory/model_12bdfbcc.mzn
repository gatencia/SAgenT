array[1..3] of var bool: x;

% --- User High-Level Code ---

        % Variables: pos_R{r}_{x}_{y}_t{t}
        var bool: pos_R0_0_0_t0; var bool: pos_R0_0_1_t1; var bool: pos_R0_1_1_t2;
        var bool: pos_R1_1_1_t0; var bool: pos_R1_1_0_t1; var bool: pos_R1_0_0_t2;
        
        % Start/Goal conditions
        constraint pos_R0_0_0_t0 = true; 
        constraint pos_R0_1_1_t2 = true;
        constraint pos_R1_1_1_t0 = true; 
        constraint pos_R1_0_0_t2 = true;
        
        % Valid Transition: (0,0)->(0,1) for R0
        constraint pos_R0_0_0_t0 -> pos_R0_0_1_t1;
        constraint pos_R0_0_1_t1 -> pos_R0_1_1_t2;
        
        % Valid Transition: (1,1)->(1,0) for R1
        constraint pos_R1_1_1_t0 -> pos_R1_1_0_t1;
        constraint pos_R1_1_0_t1 -> pos_R1_0_0_t2;
        
        % Helper Not Condition
        constraint not (pos_R0_0_1_t1 /\ pos_R1_1_0_t1 = false);
        
solve satisfy;