#-------------------------------------------------------------------------
# pyvale: single element test 3D
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

endTime = 20
timeStep = 1

# Geometric Properties
lengX = 10e-3   # m
lengY = 10e-3   # m
lengZ = 10e-3   # m

# Mesh Properties
nElemX = 2
nElemY = 2
nElemZ = 2
eType = HEX8 # TET4, TET10, TET11, HEX8, HEX20, HEX27

# Thermal BCs
coolantTemp=100.0      # degC
heatTransCoeff=125.0e3 # W.m^-2.K^-1
surfHeatFlux=4.67e6    # W.m^-2, taken from Adel's first paper
timeConst = 1   # s

# Mechanical Loads/BCs
topDispRate = ${fparse 1e-3 / endTime}  # m/s

# Thermal Props:
Density = 8829.0  # kg.m^-3
ThermCond = 384.0 # W.m^-1.K^-1
SpecHeat = 406.0  # J.kg^-1.K^-1

# Material Properties:
EMod= 100e9     # Pa
PRatio = 0.33    # -

# Thermo-mechanical coupling
stressFreeTemp = 20 # degC
ThermExp = 17.8e-6 # 1/degC

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = 3
        nx = ${nElemX}
        ny = ${nElemY}
        nz = ${nElemZ}
        xmax = ${lengX}
        ymax = ${lengY}
        zmax = ${lengZ}
        elem_type = ${eType}
    []
[]

[Variables]
    [temperature]
      family = LAGRANGE
      order = FIRST
      initial_condition = ${coolantTemp}
    []
[]

[Kernels]
    [heat_conduction]
      type = ADADHeatConduction
      variable = temperature
    []
    [time_derivative]
      type = ADADHeatConductionTimeDerivative
      variable = temperature
    []
[]


[Physics/SolidMechanics/QuasiStatic]
    [all]
        strain = SMALL
        incremental = true
        add_variables = true
        use_automatic_differentiation = true
        material_output_family = MONOMIAL   # MONOMIAL, LAGRANGE
        material_output_order = FIRST       # CONSTANT, FIRST, SECOND,
        generate_output = 'strain_xx strain_yy strain_zz strain_xy strain_yz strain_xz'
    []
[]


[BCs]
    [heat_flux_in]
        type = ADFunctionNeumannBC
        variable = temperature
        boundary = 'top'
        function = '${fparse surfHeatFlux}*(1-exp(-(1/${timeConst})*t))'
    []
    [heat_flux_out]
        type = ADConvectiveHeatFluxBC
        variable = temperature
        boundary = 'bottom'
        T_infinity = ${coolantTemp}
        heat_transfer_coefficient = ${heatTransCoeff}
    []

    [bottom_x]
        type = ADDirichletBC
        variable = disp_x
        boundary = 'bottom'
        value = 0.0
    []
    [bottom_y]
        type = ADDirichletBC
        variable = disp_y
        boundary = 'bottom'
        value = 0.0
    []
    [bottom_z]
        type = ADDirichletBC
        variable = disp_z
        boundary = 'bottom'
        value = 0.0
    []
    [top_x]
        type = ADDirichletBC
        variable = disp_x
        boundary = 'top'
        value = 0.0
    []
    [top_y]
        type = ADFunctionDirichletBC
        variable = disp_y
        boundary = 'top'
        function = '${topDispRate}*t'
    []
    [top_z]
        type = ADDirichletBC
        variable = disp_z
        boundary = 'top'
        value = 0.0
    []
[]

[Materials]
    [mat_thermal]
        type = ADADHeatConductionMaterial
        thermal_conductivity = ${ThermCond}
        specific_heat = ${SpecHeat}
    []
    [mat_density]
        type = ADADGenericConstantMaterial
        prop_names = 'density'
        prop_values = ${Density}
    []
    [mat_expansion]
        type = ADADComputeThermalExpansionEigenstrain
        temperature = temperature
        stress_free_temperature = ${stressFreeTemp}
        thermal_expansion_coeff = ${ThermExp}
        eigenstrain_name = thermal_expansion_eigenstrain
    []

    [mat_elasticity]
        type = ADComputeIsotropicElasticityTensor
        youngs_modulus = ${EMod}
        poissons_ratio = ${PRatio}
    []
    [stress]
        type =  ADComputeFiniteStrainElasticStress
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

    l_max_its = 1000
    l_tol = 1e-6

    nl_max_its = 50
    nl_rel_tol = 1e-6
    nl_abs_tol = 1e-6

    start_time=0.0
    end_time = ${endTime}
    dt = ${timeStep}

    [Predictor]
      type = SimplePredictor
      scale = 1
    []
[]


[Postprocessors]
    [react_y_bot]
        type = ADADSidesetReaction
        direction = '0 1 0'
        stress_tensor = stress
        boundary = 'bottom'
    []
    [react_y_top]
        type = ADADSidesetReaction
        direction = '0 1 0'
        stress_tensor = stress
        boundary = 'top'
    []

    [disp_y_max]
        type = NodalExtremeValue
        variable = disp_y
    []
    [disp_x_max]
        type = NodalExtremeValue
        variable = disp_x
    []
    [disp_z_max]
        type = NodalExtremeValue
        variable = disp_z
    []

[]

[Outputs]
    exodus = true
[]