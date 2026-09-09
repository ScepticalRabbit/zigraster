SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;
General.Terminal = 0;
r = 0.005;
Sphere(1) = {0, 0, 0, r};
p_top = newp; Point(p_top) = {0, r, 0};
p_bot = newp; Point(p_bot) = {0, -r, 0};
BooleanFragments{ Volume{1}; Delete; }{ Point{p_top, p_bot}; Delete; }
MeshSize{ PointsOf{ Volume{:}; } } = 0.00168;
tol = 1e-4;
ps_top() = Point In BoundingBox{-tol, r-tol, -tol, tol, r+tol, tol};
Physical Point("bc-top") = {ps_top()};
ps_bot() = Point In BoundingBox{-tol, -r-tol, -tol, tol, -r+tol, tol};
Physical Point("bc-bot") = {ps_bot()};
Physical Volume("vol") = {Volume{:}};
Mesh.ElementOrder = 2;
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 3;
