SetFactory("OpenCASCADE");
General.Terminal = 0;
Mesh.MshFileVersion = 2.2;
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh.ElementOrder = 1;
Mesh.SecondOrderIncomplete = 0;

Box(1) = {0, 0, 0, 0.01, 0.01, 0.01};
MeshSize{ PointsOf{ Volume{1}; } } = 0.006;
r = 0.005;
diam = 2 * r;
h = 1.2 * diam;
a = r * 0.35;
b = r * 0.7071067811865475;
xc = 0.016;
zc = 0.005;

p0 = newp; Point(p0) = {xc, 0, zc};
p1 = newp; Point(p1) = {xc - a, 0, zc - a};
p2 = newp; Point(p2) = {xc + a, 0, zc - a};
p3 = newp; Point(p3) = {xc + a, 0, zc + a};
p4 = newp; Point(p4) = {xc - a, 0, zc + a};

p5 = newp; Point(p5) = {xc - b, 0, zc - b};
p6 = newp; Point(p6) = {xc + b, 0, zc - b};
p7 = newp; Point(p7) = {xc + b, 0, zc + b};
p8 = newp; Point(p8) = {xc - b, 0, zc + b};

l1 = newl; Line(l1) = {p1, p2};
l2 = newl; Line(l2) = {p2, p3};
l3 = newl; Line(l3) = {p3, p4};
l4 = newl; Line(l4) = {p4, p1};

c1 = newl; Circle(c1) = {p5, p0, p6};
c2 = newl; Circle(c2) = {p6, p0, p7};
c3 = newl; Circle(c3) = {p7, p0, p8};
c4 = newl; Circle(c4) = {p8, p0, p5};

l5 = newl; Line(l5) = {p1, p5};
l6 = newl; Line(l6) = {p2, p6};
l7 = newl; Line(l7) = {p3, p7};
l8 = newl; Line(l8) = {p4, p8};

cl0 = newcl; Curve Loop(cl0) = {l1, l2, l3, l4};
s0 = news; Plane Surface(s0) = {cl0};

cl1 = newcl; Curve Loop(cl1) = {l1, l6, -c1, -l5};
s1 = news; Plane Surface(s1) = {cl1};

cl2 = newcl; Curve Loop(cl2) = {l2, l7, -c2, -l6};
s2 = news; Plane Surface(s2) = {cl2};

cl3 = newcl; Curve Loop(cl3) = {l3, l8, -c3, -l7};
s3 = news; Plane Surface(s3) = {cl3};

cl4 = newcl; Curve Loop(cl4) = {l4, l5, -c4, -l8};
s4 = news; Plane Surface(s4) = {cl4};

Transfinite Curve{l1, l2, l3, l4, c1, c2, c3, c4} = 3;
Transfinite Curve{l5, l6, l7, l8} = 3;

Transfinite Surface{s0}; Recombine Surface{s0};
Transfinite Surface{s1}; Recombine Surface{s1};
Transfinite Surface{s2}; Recombine Surface{s2};
Transfinite Surface{s3}; Recombine Surface{s3};
Transfinite Surface{s4}; Recombine Surface{s4};

Extrude{0, h, 0}{
    Surface{s0, s1, s2, s3, s4}; Layers{3}; Recombine;
}

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
