SetFactory("OpenCASCADE");
General.Terminal = 0;
plate_width = 0.025;
plate_height = 0.035;
hole_rad = plate_width / 8;
hole_loc_x = plate_width / 2;
hole_loc_y = plate_height / 2;

s1 = news; Rectangle(s1) = {0, 0, 0, plate_width, plate_height};
s2 = news; Disk(s2) = {hole_loc_x, hole_loc_y, 0, hole_rad};
BooleanDifference{ Surface{s1}; Delete; }{ Surface{s2}; Delete; }

MeshSize{ PointsOf{ Surface{:}; } } = 0.0035;
tol = 1e-4;

pc_top() = Curve In BoundingBox{
    -tol, plate_height - tol, -tol,
    plate_width + tol, plate_height + tol, tol
};
Physical Curve("bc-top") = {pc_top()};

pc_bot() = Curve In BoundingBox{
    -tol, -tol, -tol,
    plate_width + tol, tol, tol
};
Physical Curve("bc-bot") = {pc_bot()};

Physical Surface("vol") = {Surface{:}};

Mesh.ElementOrder = 2;
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 2;
