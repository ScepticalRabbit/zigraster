import numpy as np

from riley.cyth import riley as bindings


def test_speckle_builtin_is_appended_without_renumbering() -> None:
    assert bindings.FuncShaderBuiltin.constant == 0
    assert bindings.FuncShaderBuiltin.eggbox == 8
    assert bindings.FuncShaderBuiltin.speckle == 9


def test_speckle_params_defaults_and_conversion_helper() -> None:
    speckle = bindings.Speckle2DParams()

    assert speckle.seed == 0xA511E9B3
    assert speckle.cells_per_uv == (192.0, 160.0)
    assert speckle.uv_offset == (0.0, 0.0)
    assert speckle.occupancy == 0.9
    assert speckle.radius_mean == 0.45
    assert speckle.radius_jitter == 0.08
    assert speckle.edge_softness == 0.035
    assert speckle.perlin_coverage_threshold == 0.0
    assert speckle.perlin_coverage_transition_width == 0.12
    assert speckle.foreground == 0.0
    assert speckle.background == 1.0
    assert speckle.to_func_shader_params().speckle == speckle


def test_speckle_params_pass_through_mesh_conversion() -> None:
    mesh = bindings.Mesh(
        mesh_type=bindings.MeshType.tri3,
        coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        ),
        connect=np.array([[0, 1, 2]]),
        shader_type=bindings.ShaderType.func,
        uvs=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ),
        func_shader_builtin=bindings.FuncShaderBuiltin.speckle,
        func_shader_coord_mode=bindings.FuncCoordMode.uv,
        func_shader_params=bindings.Speckle2DParams(
            seed=7,
            cells_per_uv=(12.0, 10.0),
            uv_offset=(0.25, -0.5),
            occupancy=0.75,
            radius_mean=0.4,
            radius_jitter=0.05,
            edge_softness=0.02,
            perlin_coverage_threshold=0.1,
            perlin_coverage_transition_width=0.2,
            foreground=0.1,
            background=0.9,
        ).to_func_shader_params(),
    )

    assert bindings.roi_cent_over_meshes(mesh) == (0.5, 0.5, 0.0)
