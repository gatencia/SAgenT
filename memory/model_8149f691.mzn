array[1..3] of var bool: x;

% --- User High-Level Code ---
var bool: p_0; var bool: p_1; constraint p_0 != p_1;
solve satisfy;