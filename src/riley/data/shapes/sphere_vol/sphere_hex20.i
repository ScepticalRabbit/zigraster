[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    type = FileMesh
    file = 'sphere_hex20.msh'
[]

[Variables]
    [temperature]
        family = LAGRANGE
        order = SECOND
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
        material_output_order = SECOND
        generate_output = 'vonmises_stress stress_xx stress_yy stress_zz stress_xy stress_yz stress_xz strain_xx strain_yy strain_zz strain_xy strain_yz strain_xz'
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
