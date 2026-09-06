from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import netCDF4
import numpy as np

from run_helpers import extract_exodus_csvs, run_gmsh_file, run_moose_file


@dataclass(frozen=True)
class SimCase:
    shape: str
    elem_type: str
    is_pure_moose: bool
    shape_dir: str
    file_prefix: str
    geo_content: str | None
    moose_content: str


def _moose_pure_cube(elem_type: str) -> str:
    gen_outputs = (
        "vonmises_stress "
        "stress_xx stress_yy stress_zz stress_xy stress_yz stress_xz "
        "strain_xx strain_yy strain_zz strain_xy strain_yz strain_xz"
    )
    return f"""[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = 3
        nx = 2
        ny = 2
        nz = 2
        xmax = 0.01
        ymax = 0.01
        zmax = 0.01
        elem_type = {elem_type}
    []
[]

[Variables]
    [temperature]
        family = LAGRANGE
        order = FIRST
        initial_condition = 20.0
    []
[]

[Kernels]
    [heat_conduction]
        type = HeatConduction
        variable = temperature
    []
    [time_derivative]
        type = HeatConductionTimeDerivative
        variable = temperature
    []
[]

[Physics/SolidMechanics/QuasiStatic]
    [all]
        strain = SMALL
        incremental = true
        add_variables = true
        material_output_family = MONOMIAL
        material_output_order = FIRST
        generate_output = '{gen_outputs}'
    []
[]

[BCs]
    [bottom_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bottom'
        value = 0.0
    []
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = 'bottom'
        value = 0.0
    []
    [bottom_z]
        type = DirichletBC
        variable = disp_z
        boundary = 'bottom'
        value = 0.0
    []
    [top_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'top'
        value = 0.0
    []
    [top_y]
        type = FunctionDirichletBC
        variable = disp_y
        boundary = 'top'
        function = '2.5e-5*t'
    []
    [top_z]
        type = DirichletBC
        variable = disp_z
        boundary = 'top'
        value = 0.0
    []

    [heat_flux_in]
        type = FunctionNeumannBC
        variable = temperature
        boundary = 'top'
        function = '5.0e4*t'
    []
    [heat_flux_out]
        type = ConvectiveHeatFluxBC
        variable = temperature
        boundary = 'bottom'
        T_infinity = 20.0
        heat_transfer_coefficient = 1000.0
    []
[]

[Materials]
    [mat_thermal]
        type = HeatConductionMaterial
        thermal_conductivity = 16.3
        specific_heat = 500.0
    []
    [mat_density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = 8000.0
    []
    [mat_expansion]
        type = ComputeThermalExpansionEigenstrain
        temperature = temperature
        stress_free_temperature = 20.0
        thermal_expansion_coeff = 16.0e-6
        eigenstrain_name = thermal_expansion_eigenstrain
    []
    [mat_elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = 200.0e9
        poissons_ratio = 0.30
    []
    [stress]
        type = ComputeFiniteStrainElasticStress
    []
[]

[Preconditioning]
    [SMP]
        type = SMP
        full = true
    []
[]

[Executioner]
    type = Transient
    solve_type = 'NEWTON'
    petsc_options = '-snes_converged_reason'
    petsc_options_iname = '-pc_type -pc_hypre_type'
    petsc_options_value = 'hypre boomeramg'

    l_max_its = 500
    l_tol = 1e-6
    nl_max_its = 50
    nl_rel_tol = 1e-6
    nl_abs_tol = 1e-6

    start_time = 0.0
    end_time = 4.0
    dt = 1.0
[]

[Outputs]
    exodus = true
[]
"""


def _moose_gmsh_case(msh_filename: str) -> str:
    gen_outputs = (
        "vonmises_stress "
        "stress_xx stress_yy stress_zz stress_xy stress_yz stress_xz "
        "strain_xx strain_yy strain_zz strain_xy strain_yz strain_xz"
    )
    return f"""[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    type = FileMesh
    file = '{msh_filename}'
[]

[Variables]
    [temperature]
        family = LAGRANGE
        order = FIRST
        initial_condition = 20.0
    []
[]

[Kernels]
    [heat_conduction]
        type = HeatConduction
        variable = temperature
    []
    [time_derivative]
        type = HeatConductionTimeDerivative
        variable = temperature
    []
[]

[Physics/SolidMechanics/QuasiStatic]
    [all]
        strain = SMALL
        incremental = true
        add_variables = true
        material_output_family = MONOMIAL
        material_output_order = FIRST
        generate_output = '{gen_outputs}'
    []
[]

[BCs]
    [bottom_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-bot'
        value = 0.0
    []
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = 'bc-bot'
        value = 0.0
    []
    [bottom_z]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-bot'
        value = 0.0
    []
    [top_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-top'
        value = 0.0
    []
    [top_y]
        type = FunctionDirichletBC
        variable = disp_y
        boundary = 'bc-top'
        function = '2.5e-5*t'
    []
    [top_z]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-top'
        value = 0.0
    []

    [heat_flux_in]
        type = FunctionNeumannBC
        variable = temperature
        boundary = 'bc-top'
        function = '5.0e4*t'
    []
    [heat_flux_out]
        type = ConvectiveHeatFluxBC
        variable = temperature
        boundary = 'bc-bot'
        T_infinity = 20.0
        heat_transfer_coefficient = 1000.0
    []
[]

[Materials]
    [mat_thermal]
        type = HeatConductionMaterial
        thermal_conductivity = 16.3
        specific_heat = 500.0
    []
    [mat_density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = 8000.0
    []
    [mat_expansion]
        type = ComputeThermalExpansionEigenstrain
        temperature = temperature
        stress_free_temperature = 20.0
        thermal_expansion_coeff = 16.0e-6
        eigenstrain_name = thermal_expansion_eigenstrain
    []
    [mat_elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = 200.0e9
        poissons_ratio = 0.30
    []
    [stress]
        type = ComputeFiniteStrainElasticStress
    []
[]

[Preconditioning]
    [SMP]
        type = SMP
        full = true
    []
[]

[Executioner]
    type = Transient
    solve_type = 'NEWTON'
    petsc_options = '-snes_converged_reason'
    petsc_options_iname = '-pc_type -pc_hypre_type'
    petsc_options_value = 'hypre boomeramg'

    l_max_its = 500
    l_tol = 1e-6
    nl_max_its = 50
    nl_rel_tol = 1e-6
    nl_abs_tol = 1e-6

    start_time = 0.0
    end_time = 4.0
    dt = 1.0
[]

[Outputs]
    exodus = true
[]
"""


def _geo_cube_hex(elem_order: int, second_order_incomp: int) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
L = 0.01;
Box(1) = {{0, 0, 0, L, L, L}};
Transfinite Curve{{:}} = 3;
Transfinite Surface{{:}};
Recombine Surface{{:}};
Transfinite Volume{{1}};
tol = 1e-4;
ps_bot() = Surface In BoundingBox{{-tol, -tol, -tol, L+tol, tol, L+tol}};
Physical Surface("bc-bot") = {{ps_bot()}};
ps_top() = Surface In BoundingBox{{-tol, L-tol, -tol, L+tol, L+tol, L+tol}};
Physical Surface("bc-top") = {{ps_top()}};
Physical Volume("vol") = {{1}};
Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderIncomplete = {second_order_incomp};
Mesh.SecondOrderLinear = 0;
Mesh 3;
"""


def _geo_cube_tet(elem_order: int) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
L = 0.01;
Box(1) = {{0, 0, 0, L, L, L}};
MeshSize{{ PointsOf{{ Volume{{1}}; }} }} = 0.006;
tol = 1e-4;
ps_bot() = Surface In BoundingBox{{-tol, -tol, -tol, L+tol, tol, L+tol}};
Physical Surface("bc-bot") = {{ps_bot()}};
ps_top() = Surface In BoundingBox{{-tol, L-tol, -tol, L+tol, L+tol, L+tol}};
Physical Surface("bc-top") = {{ps_top()}};
Physical Volume("vol") = {{1}};
Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderLinear = 0;
Mesh 3;
"""


def _geo_cylinder_hex(elem_order: int, second_order_incomp: int) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
r = 0.005;
diam = 2 * r;
h = 1.2 * diam;
a = r * 0.35;
b = r * 0.7071067811865475;

p0 = newp; Point(p0) = {{0, 0, 0}};
p1 = newp; Point(p1) = {{-a, 0, -a}};
p2 = newp; Point(p2) = {{ a, 0, -a}};
p3 = newp; Point(p3) = {{ a, 0,  a}};
p4 = newp; Point(p4) = {{-a, 0,  a}};

p5 = newp; Point(p5) = {{-b, 0, -b}};
p6 = newp; Point(p6) = {{ b, 0, -b}};
p7 = newp; Point(p7) = {{ b, 0,  b}};
p8 = newp; Point(p8) = {{-b, 0,  b}};

l1 = newl; Line(l1) = {{p1, p2}};
l2 = newl; Line(l2) = {{p2, p3}};
l3 = newl; Line(l3) = {{p3, p4}};
l4 = newl; Line(l4) = {{p4, p1}};

c1 = newl; Circle(c1) = {{p5, p0, p6}};
c2 = newl; Circle(c2) = {{p6, p0, p7}};
c3 = newl; Circle(c3) = {{p7, p0, p8}};
c4 = newl; Circle(c4) = {{p8, p0, p5}};

l5 = newl; Line(l5) = {{p1, p5}};
l6 = newl; Line(l6) = {{p2, p6}};
l7 = newl; Line(l7) = {{p3, p7}};
l8 = newl; Line(l8) = {{p4, p8}};

cl0 = newcl; Curve Loop(cl0) = {{l1, l2, l3, l4}};
s0 = news; Plane Surface(s0) = {{cl0}};

cl1 = newcl; Curve Loop(cl1) = {{l1, l6, -c1, -l5}};
s1 = news; Plane Surface(s1) = {{cl1}};

cl2 = newcl; Curve Loop(cl2) = {{l2, l7, -c2, -l6}};
s2 = news; Plane Surface(s2) = {{cl2}};

cl3 = newcl; Curve Loop(cl3) = {{l3, l8, -c3, -l7}};
s3 = news; Plane Surface(s3) = {{cl3}};

cl4 = newcl; Curve Loop(cl4) = {{l4, l5, -c4, -l8}};
s4 = news; Plane Surface(s4) = {{cl4}};

Transfinite Curve{{l1, l2, l3, l4, c1, c2, c3, c4}} = 3;
Transfinite Curve{{l5, l6, l7, l8}} = 3;

Transfinite Surface{{s0}}; Recombine Surface{{s0}};
Transfinite Surface{{s1}}; Recombine Surface{{s1}};
Transfinite Surface{{s2}}; Recombine Surface{{s2}};
Transfinite Surface{{s3}}; Recombine Surface{{s3}};
Transfinite Surface{{s4}}; Recombine Surface{{s4}};

Extrude{{0, h, 0}}{{
    Surface{{s0, s1, s2, s3, s4}}; Layers{{3}}; Recombine;
}}

tol = 1e-4;
ps_bot() = Surface In BoundingBox{{-r-tol, -tol, -r-tol, r+tol, tol, r+tol}};
Physical Surface("bc-bot") = {{ps_bot()}};
ps_top() = Surface In BoundingBox{{-r-tol, h-tol, -r-tol, r+tol, h+tol, r+tol}};
Physical Surface("bc-top") = {{ps_top()}};
Physical Volume("vol") = {{Volume{{:}}}};

Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderIncomplete = {second_order_incomp};
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 3;
"""


def _geo_cylinder_tet(elem_order: int) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
r = 0.005;
diam = 2 * r;
h = 1.2 * diam;
v1 = newv;
Cylinder(v1) = {{0, 0, 0, 0, h, 0, r, 2*Pi}};
MeshSize{{ PointsOf{{ Volume{{v1}}; }} }} = 0.004;
tol = 1e-4;
ps_bot() = Surface In BoundingBox{{-r-tol, -tol, -r-tol, r+tol, tol, r+tol}};
Physical Surface("bc-bot") = {{ps_bot()}};
ps_top() = Surface In BoundingBox{{-r-tol, h-tol, -r-tol, r+tol, h+tol, r+tol}};
Physical Surface("bc-top") = {{ps_top()}};
Physical Volume("vol") = {{v1}};
Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 3;
"""


def _geo_platehole_hex(elem_order: int, second_order_incomp: int) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
plate_width = 25e-3;
plate_height = plate_width + 10e-3;
plate_diff = plate_height - plate_width;
plate_thick = 1e-3;

hole_rad = plate_width / 8;
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
Rectangle(s1) = {{0.0, 0.0, 0.0, plate_width / 2, plate_diff / 2}};
s2 = news;
Rectangle(s2) = {{plate_width / 2, 0.0, 0.0, plate_width / 2, plate_diff / 2}};
s3 = news;
Rectangle(s3) = {{0.0, plate_diff / 2, 0.0, plate_width / 2, plate_width / 2}};
s4 = news;
Rectangle(s4) = {{
    plate_width / 2, plate_diff / 2, 0.0, plate_width / 2, plate_width / 2
}};
s5 = news;
Rectangle(s5) = {{
    0.0, plate_width / 2 + plate_diff / 2, 0.0,
    plate_width / 2, plate_width / 2
}};
s6 = news;
Rectangle(s6) = {{
    plate_width / 2, plate_width / 2 + plate_diff / 2, 0.0,
    plate_width / 2, plate_width / 2
}};
s7 = news;
Rectangle(s7) = {{
    0.0, plate_height - plate_diff / 2, 0.0,
    plate_width / 2, plate_diff / 2
}};
s8 = news;
Rectangle(s8) = {{
    plate_width / 2, plate_height - plate_diff / 2, 0.0,
    plate_width / 2, plate_diff / 2
}};

BooleanFragments{{ Surface{{s1}}; Delete; }}
                {{ Surface{{s2, s3, s4, s5, s6, s7, s8}}; Delete; }}

c2 = newc; Circle(c2) = {{hole_loc_x, hole_loc_y, 0.0, hole_rad}};
cl2 = newcl; Curve Loop(cl2) = {{c2}};
s9 = news; Plane Surface(s9) = {{cl2}};
BooleanDifference{{ Surface{{s3, s4, s5, s6}}; Delete; }}
                  {{ Surface{{s9}}; Delete; }}

Transfinite Curve{{31, 24, 26, 28}} = plate_rad_nodes;
Transfinite Curve{{
    1, 5, 3, 7, 23, 29, 30, 34, 14, 17, 19, 22
}} = plate_edge_nodes;
Transfinite Curve{{32, 33, 25, 27}} = hole_sect_nodes;
Transfinite Curve{{4, 2, 6, 20, 18, 21}} = plate_diff_nodes;

Transfinite Surface{{s1}} = {{1, 2, 3, 4}}; Recombine Surface{{s1}};
Transfinite Surface{{s2}} = {{2, 5, 6, 3}}; Recombine Surface{{s2}};
Transfinite Surface{{s3}} = {{17, 18, 3, 16}}; Recombine Surface{{s3}};
Transfinite Surface{{s4}} = {{18, 19, 20, 3}}; Recombine Surface{{s4}};
Transfinite Surface{{s5}} = {{17, 21, 10, 16}}; Recombine Surface{{s5}};
Transfinite Surface{{s6}} = {{19, 21, 10, 20}}; Recombine Surface{{s6}};
Transfinite Surface{{s7}} = {{11, 10, 13, 14}}; Recombine Surface{{s7}};
Transfinite Surface{{s8}} = {{10, 12, 15, 13}}; Recombine Surface{{s8}};

Extrude{{0.0, 0.0, plate_thick}}{{
    Surface{{:}}; Layers{{1}}; Recombine;
}}

Physical Volume("vol") = {{Volume{{:}}}};

ps1() = Surface In BoundingBox{{
    0.0 - tol, plate_height - tol, 0.0 - tol,
    plate_width + tol, plate_height + tol, plate_thick + tol}};
Physical Surface("bc-top") = {{ps1(0), ps1(1)}};

ps2() = Surface In BoundingBox{{
    0.0 - tol, 0.0 - tol, 0.0 - tol,
    plate_width + tol, 0.0 + tol, plate_thick + tol}};
Physical Surface("bc-bot") = {{ps2(0), ps2(1)}};

Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderIncomplete = {second_order_incomp};
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 3;
"""


def _geo_platehole_tet(elem_order: int) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
plate_width = 0.025;
plate_height = 0.035;
plate_thick = 0.002;
hole_rad = plate_width / 8;
hole_loc_x = plate_width / 2;
hole_loc_y = plate_height / 2;

v1 = newv; Box(v1) = {{0, 0, 0, plate_width, plate_height, plate_thick}};
v2 = newv;
Cylinder(v2) = {{
    hole_loc_x, hole_loc_y, -0.001, 0, 0, plate_thick + 0.002, hole_rad, 2 * Pi
}};
BooleanDifference{{ Volume{{v1}}; Delete; }}{{ Volume{{v2}}; Delete; }}

MeshSize{{ PointsOf{{ Volume{{:}}; }} }} = 0.005;
tol = 1e-4;

ps_top() = Surface In BoundingBox{{
    -tol, plate_height - tol, -tol,
    plate_width + tol, plate_height + tol, plate_thick + tol}};
Physical Surface("bc-top") = {{ps_top()}};

ps_bot() = Surface In BoundingBox{{
    -tol, -tol, -tol,
    plate_width + tol, tol, plate_thick + tol}};
Physical Surface("bc-bot") = {{ps_bot()}};
Physical Volume("vol") = {{Volume{{:}}}};

Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 3;
"""


def define_all_cases() -> list[SimCase]:
    cases: list[SimCase] = []

    # Pure MOOSE Cubes (flat under data/shapes/cube/)
    for etype in ("HEX8", "HEX20", "HEX27", "TET4", "TET10"):
        file_prefix = f"cube_pure_{etype.lower()}"
        cases.append(
            SimCase(
                shape="cube",
                elem_type=etype,
                is_pure_moose=True,
                shape_dir="cube",
                file_prefix=file_prefix,
                geo_content=None,
                moose_content=_moose_pure_cube(etype),
            )
        )

    # Gmsh Cubes (flat under data/shapes/cube/)
    for etype, (order, incomp) in {
        "HEX8": (1, 0),
        "HEX20": (2, 1),
        "HEX27": (2, 0),
    }.items():
        file_prefix = f"cube_{etype.lower()}"
        cases.append(
            SimCase(
                shape="cube",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="cube",
                file_prefix=file_prefix,
                geo_content=_geo_cube_hex(order, incomp),
                moose_content=_moose_gmsh_case(f"{file_prefix}.msh"),
            )
        )
    for etype, order in {"TET4": 1, "TET10": 2}.items():
        file_prefix = f"cube_{etype.lower()}"
        cases.append(
            SimCase(
                shape="cube",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="cube",
                file_prefix=file_prefix,
                geo_content=_geo_cube_tet(order),
                moose_content=_moose_gmsh_case(f"{file_prefix}.msh"),
            )
        )

    # Gmsh Cylinders (flat under data/shapes/cylinder/)
    for etype, (order, incomp) in {
        "HEX8": (1, 0),
        "HEX20": (2, 1),
        "HEX27": (2, 0),
    }.items():
        file_prefix = f"cylinder_{etype.lower()}"
        cases.append(
            SimCase(
                shape="cylinder",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="cylinder",
                file_prefix=file_prefix,
                geo_content=_geo_cylinder_hex(order, incomp),
                moose_content=_moose_gmsh_case(f"{file_prefix}.msh"),
            )
        )
    for etype, order in {"TET4": 1, "TET10": 2}.items():
        file_prefix = f"cylinder_{etype.lower()}"
        cases.append(
            SimCase(
                shape="cylinder",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="cylinder",
                file_prefix=file_prefix,
                geo_content=_geo_cylinder_tet(order),
                moose_content=_moose_gmsh_case(f"{file_prefix}.msh"),
            )
        )

    # Gmsh Plates with Hole (flat under data/shapes/platewithhole/)
    for etype, (order, incomp) in {
        "HEX8": (1, 0),
        "HEX20": (2, 1),
        "HEX27": (2, 0),
    }.items():
        file_prefix = f"platewithhole_{etype.lower()}"
        cases.append(
            SimCase(
                shape="platewithhole",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="platewithhole",
                file_prefix=file_prefix,
                geo_content=_geo_platehole_hex(order, incomp),
                moose_content=_moose_gmsh_case(f"{file_prefix}.msh"),
            )
        )
    for etype, order in {"TET4": 1, "TET10": 2}.items():
        file_prefix = f"platewithhole_{etype.lower()}"
        cases.append(
            SimCase(
                shape="platewithhole",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="platewithhole",
                file_prefix=file_prefix,
                geo_content=_geo_platehole_tet(order),
                moose_content=_moose_gmsh_case(f"{file_prefix}.msh"),
            )
        )

    return cases


def run_all(
    shapes_root: Path,
    threads: int = 4,
) -> None:
    cases = define_all_cases()
    results: list[dict[str, object]] = []

    total_start = time.perf_counter()

    for idx, case in enumerate(cases, 1):
        target_dir = shapes_root / case.shape_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{idx}/{len(cases)}] Setting up {case.shape_dir}/"
            f"{case.file_prefix}..."
        )

        moose_file = target_dir / f"{case.file_prefix}.i"
        moose_file.write_text(case.moose_content, encoding="utf-8")

        start_time = time.perf_counter()

        if not case.is_pure_moose and case.geo_content is not None:
            geo_file = target_dir / f"{case.file_prefix}.geo"
            geo_file.write_text(case.geo_content, encoding="utf-8")
            msh_file = target_dir / f"{case.file_prefix}.msh"
            run_gmsh_file(geo_file, out_msh=msh_file, threads=threads)

        exo_file = run_moose_file(moose_file, threads=threads)
        csv_paths = extract_exodus_csvs(
            exo_file,
            out_dir=target_dir,
            prefix=f"{case.file_prefix}_",
        )

        elapsed = time.perf_counter() - start_time

        with netCDF4.Dataset(str(exo_file), "r") as ds:
            num_nodes = ds.dimensions["num_nodes"].size
            num_elem = ds.dimensions["num_elem"].size
            num_steps = ds.dimensions["time_step"].size

        results.append(
            {
                "case": case.file_prefix,
                "shape": case.shape,
                "type": case.elem_type,
                "nodes": num_nodes,
                "elements": num_elem,
                "steps": num_steps,
                "time_s": elapsed,
            }
        )

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    header = (
        f"{'Case':<28} | {'Shape':<14} | {'Type':<6} | "
        f"{'Nodes':>6} | {'Elem':>6} | {'Steps':>5} | {'Time (s)':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        line = (
            f"{str(r['case']):<28} | {str(r['shape']):<14} | "
            f"{str(r['type']):<6} | {int(r['nodes']):>6d} | "
            f"{int(r['elements']):>6d} | {int(r['steps']):>5d} | "
            f"{float(r['time_s']):>8.2f}"
        )
        print(line)
    print("=" * 80)
    print(f"Total time elapsed: {total_time:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all shape simulations and generate mesh datasets."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for Gmsh/MOOSE (default: 4)",
    )
    args = parser.parse_args()

    shapes_root = Path(__file__).resolve().parents[1]
    run_all(shapes_root, threads=args.threads)


if __name__ == "__main__":
    main()
