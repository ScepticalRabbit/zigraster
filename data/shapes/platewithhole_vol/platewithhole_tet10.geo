SetFactory("OpenCASCADE");
General.Terminal = 0;
plate_width = 0.010;
plate_height = 0.012;
plate_thick = 0.0005;
hole_rad = 0.0005;
hole_loc_x = plate_width / 2;
hole_loc_y = plate_height / 2;

v1 = newv; Box(v1) = {0, 0, 0, plate_width, plate_height, plate_thick};
v2 = newv;
Cylinder(v2) = {
    hole_loc_x, hole_loc_y, -0.001, 0, 0, plate_thick + 0.002, hole_rad, 2 * Pi
};
BooleanDifference{ Volume{v1}; Delete; }{ Volume{v2}; Delete; }

MeshSize{ PointsOf{ Volume{:}; } } = 0.0015;
tol = 1e-4;

ps_top() = Surface In BoundingBox{
    -tol, plate_height - tol, -tol,
    plate_width + tol, plate_height + tol, plate_thick + tol};
Physical Surface("bc-top") = {ps_top()};

ps_bot() = Surface In BoundingBox{
    -tol, -tol, -tol,
    plate_width + tol, tol, plate_thick + tol};
Physical Surface("bc-bot") = {ps_bot()};
Physical Volume("vol") = {Volume{:}};

Mesh.ElementOrder = 2;
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 3;
