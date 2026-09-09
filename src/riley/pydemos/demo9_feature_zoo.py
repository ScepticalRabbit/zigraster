# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
from pathlib import Path
import shutil

import numpy as np

import riley
from riley.python import sceneops

OUT_DIR = Path.cwd() / "out_riley_py" / "demo9_feature_zoo"
PIXEL_SIZE = (5.3e-6, 5.3e-6)
FOCAL_LENGTH = 50.0e-3
FRAME_INDICES = (0, 1, 2, 3)

CASES = (
    ("cube", "quad9", riley.EElemType.QUAD9, riley.MeshType.quad9),
    ("cube", "tri6", riley.EElemType.TRI6, riley.MeshType.tri6),
    ("cylinder", "quad8", riley.EElemType.QUAD8, riley.MeshType.quad8),
    ("cylinder", "tri6", riley.EElemType.TRI6, riley.MeshType.tri6),
    (
        "platewithhole",
        "quad4",
        riley.EElemType.QUAD4,
        riley.MeshType.quad4,
    ),
    ("platewithhole", "tri3", riley.EElemType.TRI3, riley.MeshType.tri3),
)

CAMERA_CASES = (
    ((1024, 1024), (0.0, 0.0, 0.0), 1, "none"),
    ((1024, 1024), (0.0, np.deg2rad(25.0), 0.0), 4, "distortion"),
    ((1024, 1229), (0.0, np.deg2rad(-28.0), 0.0), 4, "psf"),
    (
        (1229, 1024),
        (np.deg2rad(90.0), np.deg2rad(25.0), 0.0),
        4,
        "both",
    ),
    (
        (1024, 1024),
        (np.deg2rad(18.0), np.deg2rad(38.0), np.deg2rad(26.0)),
        4,
        "corner",
    ),
    (
        (1229, 1024),
        (np.deg2rad(-90.0), np.deg2rad(-20.0), np.deg2rad(5.0)),
        4,
        "ring",
    ),
)

MESH_CENTERS = (
    (-0.0125, 0.0125, 0.0),
    (0.0125, 0.0125, 0.0),
    (-0.0125, 0.0, 0.0),
    (0.0125, 0.0, 0.0),
    (-0.0125, -0.0125, 0.0),
    (0.0125, -0.0125, 0.0),
)


def load_case(shape: str, elem_type: str) -> tuple[np.ndarray, ...]:
    case_dir = riley.data.shape_surface_dataset_path(shape, elem_type)
    return (
        riley.load_csv(case_dir / "coords.csv"),
        riley.load_csv(case_dir / "connect.csv", dtype=np.int64),
        riley.load_csv(case_dir / "uvs.csv"),
        riley.load_csv(case_dir / "temperature.csv")[:, FRAME_INDICES],
        *(
            riley.load_csv(case_dir / f"disp_{axis}.csv")[:, FRAME_INDICES]
            for axis in "xyz"
        ),
    )


def make_shader(
    case_index: int,
    channels: int,
    bits: int,
    uvs: np.ndarray,
    temperature: np.ndarray,
    disp: tuple[np.ndarray, np.ndarray, np.ndarray],
    texture: np.ndarray,
) -> riley.Shader:
    normal_modes = (
        riley.NormalType.none,
        riley.NormalType.exact,
        riley.NormalType.averaged,
    )
    normal_type = normal_modes[case_index % len(normal_modes)]

    if case_index in (0, 2):
        sample = (
            riley.TextureSample.cubic_catmull_rom
            if case_index == 0
            else riley.TextureSample.linear
        )
        sample_mode = (
            riley.TextureSampleMode.lut_lerp
            if case_index == 0
            else riley.TextureSampleMode.direct
        )
        return riley.TextureShader(
            uvs=uvs,
            texture=texture,
            sample=sample,
            sample_mode=sample_mode,
            bits=bits,
            scaling_type=riley.ScaleStrategy.auto,
            normal_type=normal_type,
        )

    if case_index in (1, 5):
        field = temperature[:, :, None]
        if channels == 3:
            field = np.stack(
                (temperature, disp[0], disp[1]),
                axis=2,
            )
        return riley.NodalShader(
            field=np.ascontiguousarray(field),
            bits=bits,
            scaling_type=riley.ScaleStrategy.auto,
            scale_over=(
                riley.ScaleOver.over_frames
                if case_index == 1
                else riley.ScaleOver.within_frames
            ),
            normal_type=normal_type,
        )

    builtin = (
        riley.FuncShaderBuiltin.checker
        if case_index == 3
        else riley.FuncShaderBuiltin.eggbox
    )
    return riley.FunctionShader(
        builtin=builtin,
        coord_mode=(
            riley.FuncCoordMode.world_reference
            if case_index == 3
            else riley.FuncCoordMode.world_deformed
        ),
        params=riley.FuncShaderParams(
            coord_scale=(
                (1000.0, 1000.0)
                if case_index == 3
                else (1.0, 1.0)
            ),
            eggbox_pitch=(0.005, 0.005),
        ),
        channels=channels,
        bits=bits,
        scaling_type=riley.ScaleStrategy.auto,
        normal_type=normal_type,
    )


def build_scene(channels: int, bits: int) -> list[riley.Mesh]:
    tex_dir = riley.data.texture_dir_path()
    if (channels, bits) == (1, 8):
        texture = riley.load_texture_mono_u8(
            tex_dir / "speck128_mono_u8.bmp"
        )
    elif (channels, bits) == (1, 16):
        texture = riley.load_texture_mono_u16(
            tex_dir / "speck128_mono_u16.tiff"
        )
    elif (channels, bits) == (3, 8):
        texture = riley.load_texture_rgb_u8(
            tex_dir / "speck128_rgb_u8.bmp"
        )
    else:
        texture_u8 = riley.load_texture_rgb_u8(
            tex_dir / "speck128_rgb_u8.bmp"
        )
        texture = texture_u8.astype(np.uint16) * np.uint16(257)

    raw_data = [load_case(case[0], case[1]) for case in CASES]
    mesh_coords = [data[0] for data in raw_data]

    plate_x = np.array(mesh_coords[5][:, 0], copy=True)
    mesh_coords[5][:, 0] = -mesh_coords[5][:, 1]
    mesh_coords[5][:, 1] = plate_x

    for index, center in enumerate(MESH_CENTERS):
        sceneops.scene_center_mesh_group_at(
            mesh_coords,
            sceneops.scene_create_mesh_group_single(index),
            center,
        )

    meshes = []
    for index, (shape, elem_norm, elem_type, mesh_type) in enumerate(CASES):
        coords, connect, uvs, temperature, disp_x, disp_y, disp_z = (
            raw_data[index]
        )
        disp = (disp_x, disp_y, disp_z)
        shader = make_shader(
            index, channels, bits, uvs, temperature, disp, texture
        )
        convention = riley.ConnectConvention(
            elem_type,
            riley.EConnectAxis.ROW,
            0,
            riley.ENodeOrder.RILEY,
        )
        meshes.append(
            riley.create_mesh(
                convention,
                mesh_type,
                coords,
                connect,
                shader,
                disp=disp if index % 2 else None,
            )
        )

    return meshes


def build_cameras(meshes: list[riley.Mesh]) -> list[riley.Camera]:
    target = riley.roi_cent_over_meshes(meshes)
    base_camera = riley.Camera(
        pixels_num=(1024, 1024),
        pixels_size=PIXEL_SIZE,
        pos_world=(0.0, 0.0, 0.0),
        rot_world=(0.0, 0.0, 0.0),
        roi_cent_world=target,
        focal_length=FOCAL_LENGTH,
        sub_sample=1,
    )

    cameras = []
    for pixels_num, rotation, sub_sample, optics in CAMERA_CASES:
        position = riley.pos_frame_meshes(
            meshes,
            pixels_num,
            PIXEL_SIZE,
            FOCAL_LENGTH,
            rotation,
            fov_scale=1.1,
            target=target,
        )

        cam = copy.deepcopy(base_camera)
        cam.pixels_num = pixels_num
        cam.pos_world = position
        cam.rot_world = rotation
        cam.sub_sample = sub_sample

        if optics in ("distortion", "both", "ring"):
            cam.distortion_model = 1
            cam.distortion_k1 = -0.12
            cam.distortion_k2 = 0.035
            cam.distortion_p1 = 0.0002
            cam.distortion_p2 = -0.0001

        if optics in ("psf", "both"):
            cam.psf_type = riley.PsfType.gaussian
            cam.psf_sigma_x = 0.65
            cam.psf_sigma_y = 0.65
            cam.psf_support_rad = 2.0

        if optics == "corner":
            cam.psf_type = riley.PsfType.anisotropic_gaussian
            cam.psf_sigma_x = 0.55
            cam.psf_sigma_y = 0.9
            cam.psf_theta = float(np.deg2rad(25.0))
            cam.psf_support_rad = 2.5
            cam.psf_separable = 0

        cameras.append(cam)
    return cameras


def render_case(channels: int, bits: int) -> None:
    # --------------------------------------------------------------------------
    # 1. Build scene meshes and cameras
    # --------------------------------------------------------------------------
    meshes = build_scene(channels, bits)
    cameras = build_cameras(meshes)

    # --------------------------------------------------------------------------
    # 2. Configure raster settings and output directory
    # --------------------------------------------------------------------------
    case_name = f"{'mono' if channels == 1 else 'rgb'}-u{bits}"
    out_dir = OUT_DIR / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = riley.create_raster_config(
        num_frames=len(FRAME_INDICES),
        total_threads=4,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.image_save_mode = (
        riley.ImageSaveMode.grey if channels == 1 else riley.ImageSaveMode.rgb
    )
    config.save_bits = bits
    config.save_format = (
        riley.ImageFormat.bmp if bits == 8 else riley.ImageFormat.tiff
    )
    config.save_scaling = riley.ScaleStrategy.none
    config.background_value = 0.5 * (2**bits - 1)

    # --------------------------------------------------------------------------
    # 3. Render the multi-mesh multi-camera case
    # --------------------------------------------------------------------------
    riley.raster(meshes, cameras, config, out_dir=str(out_dir))


def main() -> None:
    # --------------------------------------------------------------------------
    # Clean output root and render all combinations
    # --------------------------------------------------------------------------
    shutil.rmtree(OUT_DIR, ignore_errors=True)

    for channels, bits in ((1, 8), (1, 16), (3, 8), (3, 16)):
        render_case(channels, bits)


if __name__ == "__main__":
    main()
