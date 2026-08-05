# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import riley

from riley.pydemos.common import make_demo_out_dir
from riley.python import sceneops


CHECKER_SQUARES_PER_AXIS = 36.0
BACKGROUND_VALUE = 127.5


def build_uv_grey_field(uvs: np.ndarray) -> np.ndarray:
    uv_scalar = 0.5 * (uvs[:, 0] + uvs[:, 1])
    return np.ascontiguousarray(uv_scalar.reshape(1, -1, 1), dtype=np.float64)


def mesh_data_name(mesh_type: riley.MeshType) -> str:
    if mesh_type in (riley.MeshType.quad4ibi, riley.MeshType.quad4newton):
        return "quad4"
    return mesh_type.name


def build_rabbit_dir(rabbit_name: str, mesh_type: riley.MeshType) -> Path:
    return riley.data.rabbit_case_path(rabbit_name, mesh_data_name(mesh_type))


def load_static_mesh(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = np.loadtxt(data_dir / "coords.csv", delimiter=",", dtype=np.float64)
    connect_float = np.loadtxt(
        data_dir / "connectivity.csv",
        delimiter=",",
        dtype=np.float64,
    )
    connect = np.ascontiguousarray(connect_float, dtype=np.uintp)
    return np.ascontiguousarray(coords, dtype=np.float64), connect


def load_uvs(data_dir: Path) -> np.ndarray:
    return np.loadtxt(data_dir / "uvs.csv", delimiter=",", dtype=np.float64)


def make_grey_mesh_input(
    mesh_type: riley.MeshType,
    mesh_idx: int,
    coords: np.ndarray,
    connect: np.ndarray,
    uvs: np.ndarray,
    texture: np.ndarray,
) -> riley.Mesh:
    shader_idx = mesh_idx % 3
    mesh_kwargs = {
        "mesh_type": mesh_type,
        "coords": np.ascontiguousarray(np.array(coords, copy=True), dtype=np.float64),
        "connect": connect,
        "bits": 8,
        "normal_type": riley.NormalType.none,
    }

    if shader_idx == 0:
        return riley.Mesh(
            shader_type=riley.ShaderType.tex,
            uvs=uvs,
            texture=texture,
            sample=riley.TextureSample.cubic_catmull_rom,
            sample_mode=riley.TextureSampleMode.lut_lerp,
            scaling_type=riley.ScaleStrategy.none,
            **mesh_kwargs,
        )

    if shader_idx == 1:
        return riley.Mesh(
            shader_type=riley.ShaderType.nodal,
            nodal_field=build_uv_grey_field(uvs),
            scaling_type=riley.ScaleStrategy.auto,
            scale_over=riley.ScaleOver.over_frames,
            **mesh_kwargs,
        )

    return riley.Mesh(
        shader_type=riley.ShaderType.func,
        uvs=uvs,
        func_shader_coord_mode=riley.FuncCoordMode.uv,
        func_shader_builtin=riley.FuncShaderBuiltin.checker,
        func_shader_params=riley.FuncShaderParams(
            coord_scale=(CHECKER_SQUARES_PER_AXIS, CHECKER_SQUARES_PER_AXIS),
        ),
        scaling_type=riley.ScaleStrategy.auto,
        **mesh_kwargs,
    )


def main() -> None:
    pixels_num = (1600, 800)
    fov_scale = 1.01
    out_dir = make_demo_out_dir("demo-rabbits")
    default_pixel_size = (5.3e-6, 5.3e-6)
    default_focal_length = 50.0e-3
    rot_world = (0.0, np.pi, 0.0)
    texture_path = riley.data.speckle_texture_path()
    rabbit_mesh_types = [
        riley.MeshType.tri3,
        riley.MeshType.tri6,
        riley.MeshType.quad4ibi,
        riley.MeshType.quad8,
        riley.MeshType.quad9,
    ]

    texture = riley.load_texture(texture_path)
    mesh_inputs: list[riley.Mesh] = []
    group_list: list[sceneops.MeshGroup] = []

    for mesh_type in rabbit_mesh_types:
        riley_dir = build_rabbit_dir("riley", mesh_type)
        feebs_dir = build_rabbit_dir("feebs", mesh_type)

        riley_coords, riley_connect = load_static_mesh(riley_dir)
        feebs_coords, feebs_connect = load_static_mesh(feebs_dir)
        riley_uvs = load_uvs(riley_dir)
        feebs_uvs = load_uvs(feebs_dir)

        pair_start = len(mesh_inputs)
        mesh_inputs.append(
            make_grey_mesh_input(
                mesh_type,
                pair_start,
                riley_coords,
                riley_connect,
                riley_uvs,
                texture,
            ),
        )
        mesh_inputs.append(
            make_grey_mesh_input(
                mesh_type,
                pair_start + 1,
                feebs_coords,
                feebs_connect,
                feebs_uvs,
                texture,
            ),
        )

        sceneops.overlap_mesh_group_bounds(
            mesh_inputs,
            sceneops.mesh_group_single(pair_start),
            sceneops.mesh_group_single(pair_start + 1),
            sceneops.BoundsOverlapSpec(
                overlap_frac=(0.85, 0.8, 0.0),
                enabled_axes=(True, True, False),
                direction=(
                    sceneops.OverlapDirection.POSITIVE,
                    sceneops.OverlapDirection.NEGATIVE,
                    sceneops.OverlapDirection.CURRENT,
                ),
            ),
        )
        group_list.append(sceneops.mesh_group_span(pair_start, 2))

    sceneops.arrange_mesh_groups_grid(
        mesh_inputs,
        group_list,
        sceneops.GridSpec(gap=(0.18, 0.28, 0.0), max_divs=(3, 2, 1)),
    )

    roi_pos = riley.roi_cent_over_meshes(mesh_inputs)
    cam_pos = riley.pos_fill_frame_from_rot_over_meshes(
        mesh_inputs,
        pixels_num,
        default_pixel_size,
        default_focal_length,
        rot_world,
        fov_scale,
    )
    camera = riley.Camera(
        pixels_num=pixels_num,
        pixels_size=default_pixel_size,
        pos_world=cam_pos,
        rot_world=rot_world,
        roi_cent_world=roi_pos,
        focal_length=default_focal_length,
        sub_sample=2,
    )

    config = riley.create_raster_config(
        num_frames=1,
        total_threads=1,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.image_save_mode = riley.ImageSaveMode.grey
    config.background_value = BACKGROUND_VALUE
    config.save_scaling = riley.ScaleStrategy.none

    start_time = perf_counter()
    riley.raster(mesh_inputs, [camera], config, out_dir=str(out_dir))
    elapsed_time = perf_counter() - start_time
    print(f"grey render time: {elapsed_time:.6f} s")
    print(f"rendered rabbits to {out_dir}")


if __name__ == "__main__":
    main()
