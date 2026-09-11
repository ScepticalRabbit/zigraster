SetFactory("OpenCASCADE");
General.Terminal = 0;
Mesh.MshFileVersion = 2.2;
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh.ElementOrder = 1;
Mesh.SecondOrderIncomplete = 0;

Box(1) = {0, 0, 0, 0.01, 0.01, 0.01};
Transfinite Curve{:} = 3;
Transfinite Surface{:};
Recombine Surface{:};
Transfinite Volume{1};
Cylinder(2) = {0.016, 0, 0.005, 0, 0.012, 0, 0.005, 2*Pi};
MeshSize{ PointsOf{ Volume{2}; } } = 0.005;

tol = 1e-4;
ps_bot() = Surface In BoundingBox{-tol, -tol, -tol, 0.03, tol, 0.02};
Physical Surface("bc-bot") = {ps_bot()};

ps_top_c() = Surface In BoundingBox{
    -tol, 0.01 - tol, -tol, 0.01 + tol, 0.01 + tol, 0.01 + tol
};
ps_top_cy() = Surface In BoundingBox{
    0.011 - tol, 0.012 - tol, -tol, 0.03, 0.012 + tol, 0.02
};
Physical Surface("bc-top") = {ps_top_c(), ps_top_cy()};

pv_cube() = Volume In BoundingBox{
    -tol, -tol, -tol, 0.01 + tol, 0.01 + tol, 0.01 + tol
};
Physical Volume("cube") = {pv_cube()};

pv_cyl() = Volume In BoundingBox{
    0.011 - tol, -tol, -tol, 0.03, 0.012 + tol, 0.02
};
Physical Volume("cylinder") = {pv_cyl()};
