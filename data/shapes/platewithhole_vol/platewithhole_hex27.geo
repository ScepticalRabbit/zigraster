SetFactory("OpenCASCADE");
General.Terminal = 0;
plate_width = 10e-3;
plate_height = 12e-3;
plate_diff = plate_height - plate_width;
plate_thick = 2.0e-3;

hole_rad = plate_width / 6;
hole_loc_x = plate_width / 2;
hole_loc_y = plate_height / 2;
hole_circ = 2 * Pi * hole_rad;

MR = 1;
hole_sect_nodes = 2 * Floor((5 * MR - 1) / 2) + 1;
plate_rad_nodes = 2 * Floor((5 * MR - 1) / 2) + 1;
plate_diff_nodes = 2 * Floor((4 * MR - 1) / 2);
plate_edge_nodes = Floor((hole_sect_nodes - 1) / 2) + 1;
elem_size = hole_circ / (4 * (hole_sect_nodes - 1));
tol = elem_size;

s1 = news;
Rectangle(s1) = {0.0, 0.0, 0.0, plate_width / 2, plate_diff / 2};
s2 = news;
Rectangle(s2) = {plate_width / 2, 0.0, 0.0, plate_width / 2, plate_diff / 2};
s3 = news;
Rectangle(s3) = {0.0, plate_diff / 2, 0.0, plate_width / 2, plate_width / 2};
s4 = news;
Rectangle(s4) = {
    plate_width / 2, plate_diff / 2, 0.0, plate_width / 2, plate_width / 2
};
s5 = news;
Rectangle(s5) = {
    0.0, plate_width / 2 + plate_diff / 2, 0.0,
    plate_width / 2, plate_width / 2
};
s6 = news;
Rectangle(s6) = {
    plate_width / 2, plate_width / 2 + plate_diff / 2, 0.0,
    plate_width / 2, plate_width / 2
};
s7 = news;
Rectangle(s7) = {
    0.0, plate_height - plate_diff / 2, 0.0,
    plate_width / 2, plate_diff / 2
};
s8 = news;
Rectangle(s8) = {
    plate_width / 2, plate_height - plate_diff / 2, 0.0,
    plate_width / 2, plate_diff / 2
};

BooleanFragments{ Surface{s1}; Delete; }
                { Surface{s2, s3, s4, s5, s6, s7, s8}; Delete; }

c2 = newc; Circle(c2) = {hole_loc_x, hole_loc_y, 0.0, hole_rad};
cl2 = newcl; Curve Loop(cl2) = {c2};
s9 = news; Plane Surface(s9) = {cl2};
BooleanDifference{ Surface{s3, s4, s5, s6}; Delete; }
                  { Surface{s9}; Delete; }

Transfinite Curve{31, 24, 26, 28} = plate_rad_nodes;
Transfinite Curve{
    1, 5, 3, 7, 23, 29, 30, 34, 14, 17, 19, 22
} = plate_edge_nodes;
Transfinite Curve{32, 33, 25, 27} = hole_sect_nodes;
Transfinite Curve{4, 2, 6, 20, 18, 21} = plate_diff_nodes;

Transfinite Surface{s1} = {1, 2, 3, 4}; Recombine Surface{s1};
Transfinite Surface{s2} = {2, 5, 6, 3}; Recombine Surface{s2};
Transfinite Surface{s3} = {17, 18, 3, 16}; Recombine Surface{s3};
Transfinite Surface{s4} = {18, 19, 20, 3}; Recombine Surface{s4};
Transfinite Surface{s5} = {17, 21, 10, 16}; Recombine Surface{s5};
Transfinite Surface{s6} = {19, 21, 10, 20}; Recombine Surface{s6};
Transfinite Surface{s7} = {11, 10, 13, 14}; Recombine Surface{s7};
Transfinite Surface{s8} = {10, 12, 15, 13}; Recombine Surface{s8};

Extrude{0.0, 0.0, plate_thick}{
    Surface{:}; Layers{1}; Recombine;
}

Physical Volume("vol") = {Volume{:}};

ps1() = Surface In BoundingBox{
    0.0 - tol, plate_height - tol, 0.0 - tol,
    plate_width + tol, plate_height + tol, plate_thick + tol};
Physical Surface("bc-top") = {ps1(0), ps1(1)};

ps2() = Surface In BoundingBox{
    0.0 - tol, 0.0 - tol, 0.0 - tol,
    plate_width + tol, 0.0 + tol, plate_thick + tol};
Physical Surface("bc-bot") = {ps2(0), ps2(1)};

Mesh.ElementOrder = 2;
Mesh.SecondOrderIncomplete = 0;
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 3;
