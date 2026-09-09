from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import netCDF4
import numpy as np

import riley
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
    dim: int = 3


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
        function = '(1.0e-3/3.0)*t'
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
    end_time = 3.0
    dt = 1.0
[]

[Outputs]
    exodus = true
[]
"""


def _moose_gmsh_case(
    msh_filename: str,
    order_str: str = "FIRST",
) -> str:
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
        order = {order_str}
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
        material_output_order = {order_str}
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
        function = '(1.0e-3/3.0)*t'
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
    end_time = 3.0
    dt = 1.0
[]

[Outputs]
    exodus = true
[]
"""


def _moose_sphere_case(
    msh_filename: str,
    order_str: str = "FIRST",
) -> str:
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
        order = {order_str}
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
        material_output_order = {order_str}
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
        function = '-(1.0e-3/3.0)*t'
    []
    [top_z]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-top'
        value = 0.0
    []

    [heat_in]
        type = FunctionDirichletBC
        variable = temperature
        boundary = 'bc-top'
        function = '20.0 + (80.0/3.0)*t'
    []
    [heat_out]
        type = DirichletBC
        variable = temperature
        boundary = 'bc-bot'
        value = 20.0
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
    end_time = 3.0
    dt = 1.0
[]

[Outputs]
    exodus = true
[]
"""


def _moose_2d_case(msh_filename: str, order_str: str) -> str:
    gen_outputs = (
        "vonmises_stress "
        "stress_xx stress_yy stress_xy "
        "strain_xx strain_yy strain_xy"
    )
    return f"""[GlobalParams]
    displacements = 'disp_x disp_y'
[]

[Mesh]
    type = FileMesh
    file = '{msh_filename}'
[]

[Variables]
    [temperature]
        family = LAGRANGE
        order = {order_str}
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
        material_output_order = {order_str}
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
        function = '(1.0e-3/3.0)*t'
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
    end_time = 3.0
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

Transfinite Curve{{l1, l2, l3, l4, c1, c2, c3, c4}} = 5;
Transfinite Curve{{l5, l6, l7, l8}} = 5;

Transfinite Surface{{s0}}; Recombine Surface{{s0}};
Transfinite Surface{{s1}}; Recombine Surface{{s1}};
Transfinite Surface{{s2}}; Recombine Surface{{s2}};
Transfinite Surface{{s3}}; Recombine Surface{{s3}};
Transfinite Surface{{s4}}; Recombine Surface{{s4}};

Extrude{{0, h, 0}}{{
    Surface{{s0, s1, s2, s3, s4}}; Layers{{6}}; Recombine;
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
MeshSize{{ PointsOf{{ Volume{{v1}}; }} }} = 0.002;
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


def _geo_sphere_hex(elem_order: int, second_order_incomp: int) -> str:
    r_val = 0.005
    a_val = 0.0018
    d_val = r_val / np.sqrt(3)
    m_val = r_val / np.sqrt(2)

    lines: list[str] = [
        "Mesh.MshFileVersion = 2.2;",
        "General.Terminal = 0;",
        f"R = {r_val};",
        f"a = {a_val};",
        "d = R / Sqrt(3);",
        "m = R / Sqrt(2);",
        "Point(1) = {0, 0, 0};",
    ]

    pts: dict[tuple[float, ...], tuple[int, str]] = {}
    pt_id = 2

    def get_pid(coords: tuple[float, ...], expr: str) -> int:
        nonlocal pt_id
        key = tuple(np.round(coords, 8))
        if key in pts:
            return pts[key][0]
        pid = pt_id
        pt_id += 1
        pts[key] = (pid, expr)
        return pid

    inner_pids = np.zeros((3, 3, 3), dtype=int)
    for ix, vx in enumerate([-a_val, 0.0, a_val]):
        for iy, vy in enumerate([-a_val, 0.0, a_val]):
            for iz, vz in enumerate([-a_val, 0.0, a_val]):
                sx = "-a" if vx < -1e-6 else ("a" if vx > 1e-6 else "0")
                sy = "-a" if vy < -1e-6 else ("a" if vy > 1e-6 else "0")
                sz = "-a" if vz < -1e-6 else ("a" if vz > 1e-6 else "0")
                inner_pids[ix, iy, iz] = get_pid(
                    (vx, vy, vz), f"{{{sx}, {sy}, {sz}}}"
                )

    outer_pids = np.zeros((3, 3, 3), dtype=int)
    for ix, sx_val in enumerate([-1, 0, 1]):
        for iy, sy_val in enumerate([-1, 0, 1]):
            for iz, sz_val in enumerate([-1, 0, 1]):
                if sx_val == 0 and sy_val == 0 and sz_val == 0:
                    continue
                nnz = abs(sx_val) + abs(sy_val) + abs(sz_val)
                if nnz == 1:
                    vx = sx_val * r_val
                    vy = sy_val * r_val
                    vz = sz_val * r_val
                    sx = f"{sx_val}*R" if sx_val != 0 else "0"
                    sy = f"{sy_val}*R" if sy_val != 0 else "0"
                    sz = f"{sz_val}*R" if sz_val != 0 else "0"
                elif nnz == 2:
                    vx = sx_val * m_val
                    vy = sy_val * m_val
                    vz = sz_val * m_val
                    sx = f"{sx_val}*m" if sx_val != 0 else "0"
                    sy = f"{sy_val}*m" if sy_val != 0 else "0"
                    sz = f"{sz_val}*m" if sz_val != 0 else "0"
                else:
                    vx = sx_val * d_val
                    vy = sy_val * d_val
                    vz = sz_val * d_val
                    sx = f"{sx_val}*d" if sx_val != 0 else "0"
                    sy = f"{sy_val}*d" if sy_val != 0 else "0"
                    sz = f"{sz_val}*d" if sz_val != 0 else "0"
                outer_pids[ix, iy, iz] = get_pid(
                    (vx, vy, vz), f"{{{sx}, {sy}, {sz}}}"
                )

    for key, (pid, expr) in pts.items():
        lines.append(f"Point({pid}) = {expr};")

    curves: dict[tuple[int, int], int] = {}
    curve_id = 1

    def get_line(p1: int, p2: int) -> tuple[int, int]:
        nonlocal curve_id
        if (p1, p2) in curves:
            return curves[(p1, p2)], 1
        if (p2, p1) in curves:
            return curves[(p2, p1)], -1
        cid = curve_id
        curve_id += 1
        curves[(p1, p2)] = cid
        lines.append(f"Line({cid}) = {{{p1}, {p2}}};")
        return cid, 1

    def get_circle(p1: int, p2: int) -> tuple[int, int]:
        nonlocal curve_id
        if (p1, p2) in curves:
            return curves[(p1, p2)], 1
        if (p2, p1) in curves:
            return curves[(p2, p1)], -1
        cid = curve_id
        curve_id += 1
        curves[(p1, p2)] = cid
        lines.append(f"Circle({cid}) = {{{p1}, 1, {p2}}};")
        return cid, 1

    surfaces: dict[tuple[int, ...], int] = {}
    surf_id = 1

    def get_surface(
        p1: int,
        p2: int,
        p3: int,
        p4: int,
        is_spherical: bool = False,
        is_plane: bool = True,
    ) -> int:
        nonlocal surf_id
        canon = tuple(sorted([p1, p2, p3, p4]))
        if canon in surfaces:
            return surfaces[canon]
        sid = surf_id
        surf_id += 1
        surfaces[canon] = sid
        c_edges = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
        clist: list[str] = []
        for ep1, ep2 in c_edges:
            cid, sign = (
                get_circle(ep1, ep2) if is_spherical else get_line(ep1, ep2)
            )
            clist.append(f"{cid}" if sign == 1 else f"-{cid}")
        loop_s = ", ".join(clist)
        lines.append(f"Line Loop({sid}) = {{{loop_s}}};")
        if is_plane:
            lines.append(f"Plane Surface({sid}) = {{{sid}}};")
        else:
            lines.append(f"Surface({sid}) = {{{sid}}};")
        return sid

    volumes: list[int] = []
    vol_id = 1

    def add_hex_block(
        corners_bot: list[int],
        corners_top: list[int],
        is_top_spherical: bool = False,
        is_bot_spherical: bool = False,
    ) -> int:
        nonlocal vol_id
        b1, b2, b3, b4 = corners_bot
        t1, t2, t3, t4 = corners_top
        s_bot = get_surface(
            b1,
            b4,
            b3,
            b2,
            is_spherical=is_bot_spherical,
            is_plane=not is_bot_spherical,
        )
        s_top = get_surface(
            t1,
            t2,
            t3,
            t4,
            is_spherical=is_top_spherical,
            is_plane=not is_top_spherical,
        )
        s_f = get_surface(
            b1, b2, t2, t1, is_spherical=False, is_plane=True
        )
        s_r = get_surface(
            b2, b3, t3, t2, is_spherical=False, is_plane=True
        )
        s_b = get_surface(
            b3, b4, t4, t3, is_spherical=False, is_plane=True
        )
        s_l = get_surface(
            b4, b1, t1, t4, is_spherical=False, is_plane=True
        )
        vid = vol_id
        vol_id += 1
        lines.append(
            f"Surface Loop({vid}) = "
            f"{{{s_bot}, {s_top}, {s_f}, {s_r}, {s_b}, {s_l}}};"
        )
        lines.append(f"Volume({vid}) = {{{vid}}};")
        volumes.append(vid)
        return vid

    # 1. 8 Inner cube blocks:
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                bot = [
                    int(inner_pids[ix, iy, iz]),
                    int(inner_pids[ix + 1, iy, iz]),
                    int(inner_pids[ix + 1, iy, iz + 1]),
                    int(inner_pids[ix, iy, iz + 1]),
                ]
                top = [
                    int(inner_pids[ix, iy + 1, iz]),
                    int(inner_pids[ix + 1, iy + 1, iz]),
                    int(inner_pids[ix + 1, iy + 1, iz + 1]),
                    int(inner_pids[ix, iy + 1, iz + 1]),
                ]
                add_hex_block(bot, top)

    # 2. 24 Outer spherical blocks:
    # Face +Y (top cap):
    for ix in range(2):
        for iz in range(2):
            bot = [
                int(inner_pids[ix, 2, iz]),
                int(inner_pids[ix + 1, 2, iz]),
                int(inner_pids[ix + 1, 2, iz + 1]),
                int(inner_pids[ix, 2, iz + 1]),
            ]
            top = [
                int(outer_pids[ix, 2, iz]),
                int(outer_pids[ix + 1, 2, iz]),
                int(outer_pids[ix + 1, 2, iz + 1]),
                int(outer_pids[ix, 2, iz + 1]),
            ]
            add_hex_block(bot, top, is_top_spherical=True)

    # Face -Y (bot cap):
    for ix in range(2):
        for iz in range(2):
            bot = [
                int(outer_pids[ix, 0, iz]),
                int(outer_pids[ix + 1, 0, iz]),
                int(outer_pids[ix + 1, 0, iz + 1]),
                int(outer_pids[ix, 0, iz + 1]),
            ]
            top = [
                int(inner_pids[ix, 0, iz]),
                int(inner_pids[ix + 1, 0, iz]),
                int(inner_pids[ix + 1, 0, iz + 1]),
                int(inner_pids[ix, 0, iz + 1]),
            ]
            add_hex_block(bot, top, is_bot_spherical=True)

    # Face +Z:
    for ix in range(2):
        for iy in range(2):
            bot = [
                int(inner_pids[ix, iy, 2]),
                int(inner_pids[ix + 1, iy, 2]),
                int(inner_pids[ix + 1, iy + 1, 2]),
                int(inner_pids[ix, iy + 1, 2]),
            ]
            top = [
                int(outer_pids[ix, iy, 2]),
                int(outer_pids[ix + 1, iy, 2]),
                int(outer_pids[ix + 1, iy + 1, 2]),
                int(outer_pids[ix, iy + 1, 2]),
            ]
            add_hex_block(bot, top, is_top_spherical=True)

    # Face -Z:
    for ix in range(2):
        for iy in range(2):
            bot = [
                int(outer_pids[ix, iy, 0]),
                int(outer_pids[ix + 1, iy, 0]),
                int(outer_pids[ix + 1, iy + 1, 0]),
                int(outer_pids[ix, iy + 1, 0]),
            ]
            top = [
                int(inner_pids[ix, iy, 0]),
                int(inner_pids[ix + 1, iy, 0]),
                int(inner_pids[ix + 1, iy + 1, 0]),
                int(inner_pids[ix, iy + 1, 0]),
            ]
            add_hex_block(bot, top, is_bot_spherical=True)

    # Face +X:
    for iy in range(2):
        for iz in range(2):
            bot = [
                int(inner_pids[2, iy, iz]),
                int(inner_pids[2, iy + 1, iz]),
                int(inner_pids[2, iy + 1, iz + 1]),
                int(inner_pids[2, iy, iz + 1]),
            ]
            top = [
                int(outer_pids[2, iy, iz]),
                int(outer_pids[2, iy + 1, iz]),
                int(outer_pids[2, iy + 1, iz + 1]),
                int(outer_pids[2, iy, iz + 1]),
            ]
            add_hex_block(bot, top, is_top_spherical=True)

    # Face -X:
    for iy in range(2):
        for iz in range(2):
            bot = [
                int(outer_pids[0, iy, iz]),
                int(outer_pids[0, iy + 1, iz]),
                int(outer_pids[0, iy + 1, iz + 1]),
                int(outer_pids[0, iy, iz + 1]),
            ]
            top = [
                int(inner_pids[0, iy, iz]),
                int(inner_pids[0, iy + 1, iz]),
                int(inner_pids[0, iy + 1, iz + 1]),
                int(inner_pids[0, iy, iz + 1]),
            ]
            add_hex_block(bot, top, is_bot_spherical=True)

    lines.append("Transfinite Curve{:} = 4;")
    lines.append("Transfinite Surface{:};")
    lines.append("Recombine Surface{:};")
    lines.append("Transfinite Volume{:};")

    top_pole_pid = outer_pids[1, 2, 1]
    bot_pole_pid = outer_pids[1, 0, 1]
    lines.append(f'Physical Point("bc-top") = {{{top_pole_pid}}};')
    lines.append(f'Physical Point("bc-bot") = {{{bot_pole_pid}}};')
    lines.append('Physical Volume("vol") = {Volume{:}};\n')

    lines.append(f"Mesh.ElementOrder = {elem_order};")
    lines.append(f"Mesh.SecondOrderIncomplete = {second_order_incomp};")
    lines.append("Mesh.SecondOrderLinear = 0;")
    lines.append("Mesh.HighOrderOptimize = 1;")
    lines.append("Mesh 3;")

    return "\n".join(lines)


def _geo_sphere_tet(elem_order: int) -> str:
    return f"""SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;
General.Terminal = 0;
r = 0.005;
Sphere(1) = {{0, 0, 0, r}};
p_top = newp; Point(p_top) = {{0, r, 0}};
p_bot = newp; Point(p_bot) = {{0, -r, 0}};
BooleanFragments{{ Volume{{1}}; Delete; }}{{ Point{{p_top, p_bot}}; Delete; }}
MeshSize{{ PointsOf{{ Volume{{:}}; }} }} = 0.00168;
tol = 1e-4;
ps_top() = Point In BoundingBox{{-tol, r-tol, -tol, tol, r+tol, tol}};
Physical Point("bc-top") = {{ps_top()}};
ps_bot() = Point In BoundingBox{{-tol, -r-tol, -tol, tol, -r+tol, tol}};
Physical Point("bc-bot") = {{ps_bot()}};
Physical Volume("vol") = {{Volume{{:}}}};
Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 3;
"""


def _geo_platehole_hex(elem_order: int, second_order_incomp: int) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
plate_width = 10e-3;
plate_height = 12e-3;
plate_diff = plate_height - plate_width;
plate_thick = 0.5e-3;

hole_rad = 0.5e-3;
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
plate_width = 0.010;
plate_height = 0.012;
plate_thick = 0.0005;
hole_rad = 0.0005;
hole_loc_x = plate_width / 2;
hole_loc_y = plate_height / 2;

v1 = newv; Box(v1) = {{0, 0, 0, plate_width, plate_height, plate_thick}};
v2 = newv;
Cylinder(v2) = {{
    hole_loc_x, hole_loc_y, -0.001, 0, 0, plate_thick + 0.002, hole_rad, 2 * Pi
}};
BooleanDifference{{ Volume{{v1}}; Delete; }}{{ Volume{{v2}}; Delete; }}

MeshSize{{ PointsOf{{ Volume{{:}}; }} }} = 0.0015;
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


def _geo_platewithhole2d_quad(
    elem_order: int,
    second_order_incomp: int,
) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
plate_width = 0.010;
plate_height = 0.012;
plate_diff = plate_height - plate_width;
hole_rad = 0.0005;
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

Physical Surface("vol") = {{Surface{{:}}}};

pc_top() = Curve In BoundingBox{{
    0.0 - tol, plate_height - tol, -tol,
    plate_width + tol, plate_height + tol, tol
}};
Physical Curve("bc-top") = {{pc_top()}};

pc_bot() = Curve In BoundingBox{{
    0.0 - tol, 0.0 - tol, -tol,
    plate_width + tol, 0.0 + tol, tol
}};
Physical Curve("bc-bot") = {{pc_bot()}};

Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderIncomplete = {second_order_incomp};
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 2;
"""


def _geo_platewithhole2d_tri(elem_order: int) -> str:
    return f"""SetFactory("OpenCASCADE");
General.Terminal = 0;
plate_width = 0.010;
plate_height = 0.012;
hole_rad = 0.0005;
hole_loc_x = plate_width / 2;
hole_loc_y = plate_height / 2;

s1 = news; Rectangle(s1) = {{0, 0, 0, plate_width, plate_height}};
s2 = news; Disk(s2) = {{hole_loc_x, hole_loc_y, 0, hole_rad}};
BooleanDifference{{ Surface{{s1}}; Delete; }}{{ Surface{{s2}}; Delete; }}

MeshSize{{ PointsOf{{ Surface{{:}}; }} }} = 0.0012;
tol = 1e-4;

pc_top() = Curve In BoundingBox{{
    -tol, plate_height - tol, -tol,
    plate_width + tol, plate_height + tol, tol
}};
Physical Curve("bc-top") = {{pc_top()}};

pc_bot() = Curve In BoundingBox{{
    -tol, -tol, -tol,
    plate_width + tol, tol, tol
}};
Physical Curve("bc-bot") = {{pc_bot()}};

Physical Surface("vol") = {{Surface{{:}}}};

Mesh.ElementOrder = {elem_order};
Mesh.SecondOrderLinear = 0;
Mesh.HighOrderOptimize = 1;
Mesh 2;
"""


def define_all_cases() -> list[SimCase]:
    cases: list[SimCase] = []

    # Pure MOOSE Cubes (under cube_vol/)
    for etype in ("HEX8", "HEX20", "HEX27", "TET4", "TET10"):
        file_prefix = f"cube_pure_{etype.lower()}"
        cases.append(
            SimCase(
                shape="cube",
                elem_type=etype,
                is_pure_moose=True,
                shape_dir="cube_vol",
                file_prefix=file_prefix,
                geo_content=None,
                moose_content=_moose_pure_cube(etype),
            )
        )

    # Gmsh Cubes (under cube_vol/)
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
                shape_dir="cube_vol",
                file_prefix=file_prefix,
                geo_content=_geo_cube_hex(order, incomp),
                moose_content=_moose_gmsh_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
            )
        )
    for etype, order in {"TET4": 1, "TET10": 2}.items():
        file_prefix = f"cube_{etype.lower()}"
        cases.append(
            SimCase(
                shape="cube",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="cube_vol",
                file_prefix=file_prefix,
                geo_content=_geo_cube_tet(order),
                moose_content=_moose_gmsh_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
            )
        )

    # Gmsh Cylinders (under cylinder_vol/)
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
                shape_dir="cylinder_vol",
                file_prefix=file_prefix,
                geo_content=_geo_cylinder_hex(order, incomp),
                moose_content=_moose_gmsh_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
            )
        )
    for etype, order in {"TET4": 1, "TET10": 2}.items():
        file_prefix = f"cylinder_{etype.lower()}"
        cases.append(
            SimCase(
                shape="cylinder",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="cylinder_vol",
                file_prefix=file_prefix,
                geo_content=_geo_cylinder_tet(order),
                moose_content=_moose_gmsh_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
            )
        )

    # Gmsh Spheres (under sphere_vol/)
    for etype, (order, incomp) in {
        "HEX8": (1, 0),
        "HEX20": (2, 1),
        "HEX27": (2, 0),
    }.items():
        file_prefix = f"sphere_{etype.lower()}"
        cases.append(
            SimCase(
                shape="sphere",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="sphere_vol",
                file_prefix=file_prefix,
                geo_content=_geo_sphere_hex(order, incomp),
                moose_content=_moose_sphere_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
            )
        )
    for etype, order in {"TET4": 1, "TET10": 2}.items():
        file_prefix = f"sphere_{etype.lower()}"
        cases.append(
            SimCase(
                shape="sphere",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="sphere_vol",
                file_prefix=file_prefix,
                geo_content=_geo_sphere_tet(order),
                moose_content=_moose_sphere_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
            )
        )

    # Gmsh Plates with Hole 3D (under platewithhole_vol/)
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
                shape_dir="platewithhole_vol",
                file_prefix=file_prefix,
                geo_content=_geo_platehole_hex(order, incomp),
                moose_content=_moose_gmsh_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
            )
        )
    for etype, order in {"TET4": 1, "TET10": 2}.items():
        file_prefix = f"platewithhole_{etype.lower()}"
        cases.append(
            SimCase(
                shape="platewithhole",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="platewithhole_vol",
                file_prefix=file_prefix,
                geo_content=_geo_platehole_tet(order),
                moose_content=_moose_gmsh_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
            )
        )

    # 2D Plate with Hole (under platewithhole_2d/)
    for etype, (order, incomp) in {
        "QUAD4": (1, 0),
        "QUAD8": (2, 1),
        "QUAD9": (2, 0),
    }.items():
        file_prefix = f"platewithhole2d_{etype.lower()}"
        cases.append(
            SimCase(
                shape="platewithhole2d",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="platewithhole_2d",
                file_prefix=file_prefix,
                geo_content=_geo_platewithhole2d_quad(order, incomp),
                moose_content=_moose_2d_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
                dim=2,
            )
        )
    for etype, order in {"TRI3": 1, "TRI6": 2}.items():
        file_prefix = f"platewithhole2d_{etype.lower()}"
        cases.append(
            SimCase(
                shape="platewithhole2d",
                elem_type=etype,
                is_pure_moose=False,
                shape_dir="platewithhole_2d",
                file_prefix=file_prefix,
                geo_content=_geo_platewithhole2d_tri(order),
                moose_content=_moose_2d_case(
                    f"{file_prefix}.msh",
                    order_str="SECOND" if order == 2 else "FIRST",
                ),
                dim=2,
            )
        )

    return cases


def extract_surface_datasets(shapes_root: Path) -> None:
    """Extract surface datasets for all shapes into *_surf directories."""
    surf_mapping = (
        (
            "cube",
            "cube_vol",
            "cube_surf",
            (
                (
                    "tri3",
                    "cube_tet4",
                    riley.EElemType.TET4,
                    riley.MeshType.tri3,
                ),
                (
                    "tri6",
                    "cube_tet10",
                    riley.EElemType.TET10,
                    riley.MeshType.tri6,
                ),
                (
                    "quad4",
                    "cube_hex8",
                    riley.EElemType.HEX8,
                    riley.MeshType.quad4,
                ),
                (
                    "quad8",
                    "cube_hex20",
                    riley.EElemType.HEX20,
                    riley.MeshType.quad8,
                ),
                (
                    "quad9",
                    "cube_hex27",
                    riley.EElemType.HEX27,
                    riley.MeshType.quad9,
                ),
            ),
        ),
        (
            "cylinder",
            "cylinder_vol",
            "cylinder_surf",
            (
                (
                    "tri3",
                    "cylinder_tet4",
                    riley.EElemType.TET4,
                    riley.MeshType.tri3,
                ),
                (
                    "tri6",
                    "cylinder_tet10",
                    riley.EElemType.TET10,
                    riley.MeshType.tri6,
                ),
                (
                    "quad4",
                    "cylinder_hex8",
                    riley.EElemType.HEX8,
                    riley.MeshType.quad4,
                ),
                (
                    "quad8",
                    "cylinder_hex20",
                    riley.EElemType.HEX20,
                    riley.MeshType.quad8,
                ),
                (
                    "quad9",
                    "cylinder_hex27",
                    riley.EElemType.HEX27,
                    riley.MeshType.quad9,
                ),
            ),
        ),
        (
            "sphere",
            "sphere_vol",
            "sphere_surf",
            (
                (
                    "tri3",
                    "sphere_tet4",
                    riley.EElemType.TET4,
                    riley.MeshType.tri3,
                ),
                (
                    "tri6",
                    "sphere_tet10",
                    riley.EElemType.TET10,
                    riley.MeshType.tri6,
                ),
                (
                    "quad4",
                    "sphere_hex8",
                    riley.EElemType.HEX8,
                    riley.MeshType.quad4,
                ),
                (
                    "quad8",
                    "sphere_hex20",
                    riley.EElemType.HEX20,
                    riley.MeshType.quad8,
                ),
                (
                    "quad9",
                    "sphere_hex27",
                    riley.EElemType.HEX27,
                    riley.MeshType.quad9,
                ),
            ),
        ),
        (
            "platewithhole",
            "platewithhole_vol",
            "platewithhole_surf",
            (
                (
                    "tri3",
                    "platewithhole_tet4",
                    riley.EElemType.TET4,
                    riley.MeshType.tri3,
                ),
                (
                    "tri6",
                    "platewithhole_tet10",
                    riley.EElemType.TET10,
                    riley.MeshType.tri6,
                ),
                (
                    "quad4",
                    "platewithhole_hex8",
                    riley.EElemType.HEX8,
                    riley.MeshType.quad4,
                ),
                (
                    "quad8",
                    "platewithhole_hex20",
                    riley.EElemType.HEX20,
                    riley.MeshType.quad8,
                ),
                (
                    "quad9",
                    "platewithhole_hex27",
                    riley.EElemType.HEX27,
                    riley.MeshType.quad9,
                ),
            ),
        ),
    )

    for shape_name, vol_dir_name, surf_dir_name, surf_specs in surf_mapping:
        vol_dir = shapes_root / vol_dir_name
        surf_dir = shapes_root / surf_dir_name
        surf_dir.mkdir(parents=True, exist_ok=True)

        for surf_type_name, vol_prefix, source_elem, mesh_type in surf_specs:
            target_sub = surf_dir / surf_type_name
            target_sub.mkdir(parents=True, exist_ok=True)

            coords_path = vol_dir / f"{vol_prefix}_coords.csv"
            connect_path = vol_dir / f"{vol_prefix}_connectivity.csv"
            coords = riley.load_csv(coords_path)
            connect = riley.load_csv(connect_path, dtype=np.int64)

            disp = tuple(
                riley.load_csv(vol_dir / f"{vol_prefix}_disp_{axis}.csv")
                for axis in "xyz"
            )
            temp = riley.load_csv(vol_dir / f"{vol_prefix}_temperature.csv")

            convention = riley.ConnectConvention(
                source_elem,
                riley.EConnectAxis.ROW,
                0,
                riley.ENodeOrder.RILEY,
            )
            mesh = riley.create_mesh(
                convention,
                mesh_type,
                coords,
                connect,
                riley.NodalShader(temp),
                disp=disp,
            )
            extent_max = max(
                float(np.ptp(mesh.coords[:, 0])),
                float(np.ptp(mesh.coords[:, 1])),
            )
            uv_span_max = (extent_max * 2500.0) / 128.0
            uvs = riley.project_uvs_planar_centered(
                mesh.coords,
                (128, 128),
                uv_span_max=uv_span_max,
                proj_plane=riley.EUVProjPlane.XY,
            )

            np.savetxt(
                target_sub / "coords.csv",
                mesh.coords,
                delimiter=",",
                fmt="%.17g",
            )
            np.savetxt(
                target_sub / "connect.csv",
                mesh.connect,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                target_sub / "uvs.csv",
                uvs,
                delimiter=",",
                fmt="%.17g",
            )
            np.savetxt(
                target_sub / "temperature.csv",
                mesh.shader.field[:, :, 0].T,
                delimiter=",",
                fmt="%.17g",
            )
            if mesh.disp is not None:
                for ax_idx, axis in enumerate("xyz"):
                    np.savetxt(
                        target_sub / f"disp_{axis}.csv",
                        mesh.disp[:, :, ax_idx].T,
                        delimiter=",",
                        fmt="%.17g",
                    )
            print(
                f"Extracted surface: {surf_dir_name}/{surf_type_name} "
                f"({mesh.coords.shape[0]} nodes, {mesh.connect.shape[0]} elems)"
            )


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

    print("\nExtracting surface meshes for all shapes...")
    extract_surface_datasets(shapes_root)

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
