[GlobalParams]
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
        elem_type = HEX8
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
        generate_output = 'vonmises_stress stress_xx stress_yy stress_zz stress_xy stress_yz stress_xz strain_xx strain_yy strain_zz strain_xy strain_yz strain_xz'
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
