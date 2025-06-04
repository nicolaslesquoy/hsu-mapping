// Number of elements in each direction
n = $nb_elements$;
// Characteristic length
lc = 1.0/n;

Point(1) = {0, 0, 0, lc};
Point(2) = {1.0, 0,  0, lc};
Point(3) = {1.0, 1.0, 0, lc};
Point(4) = {0, 1.0, 0, lc};

Line(1) = {1, 2};
Line(2) = {3, 2};
Line(3) = {3, 4};
Line(4) = {4, 1};

Transfinite Line{1, 3} = n;
Transfinite Line{2, 4} = n;

Curve Loop(1) = {4, 1, -2, 3};

Plane Surface(1) = {1};

Physical Curve(5) = {1, 2, 4};
Physical Surface("My surface") = {1};