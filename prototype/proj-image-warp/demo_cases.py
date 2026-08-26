# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Verification cases for projected 2D image warping against 3D rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import riley
from warp_engine import (
    calc_rot_matrix_zyx,
    execute_projected_warp,
    project_world_to_cam_pixels,
    render_projected_coord_map,
    warp_reference_image,
)

OUT_DIR = Path("./out/prototype_proj_image_warp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_flat_plate_mesh(
    width: float = 100.0,
    height: float = 100.0,
    divisions_x: int = 10,
    divisions_y: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a flat 2D quad4 mesh in the XY plane centered at the origin."""
    grid_x = np.linspace(-0.5 * width, 0.5 * width, divisions_x + 1)
    grid_y = np.linspace(-0.5 * height, 0.5 * height, divisions_y + 1)
    mesh_xx, mesh_yy = np.meshgrid(grid_x, grid_y)

    total_nodes = (divisions_x + 1) * (divisions_y + 1)
    coords = np.zeros((total_nodes, 3), dtype=np.float64)
    coords[:, 0] = mesh_xx.ravel()
    coords[:, 1] = mesh_yy.ravel()
    coords[:, 2] = 0.0

    # Connectivity for QUAD4: counter-clockwise
    # (bottom-left, bottom-right, top-right, top-left)
    elems_num = divisions_x * divisions_y
    connect = np.zeros((elems_num, 4), dtype=np.uint32)

    elem_idx = 0
    stride_y = divisions_x + 1
    for row_idx in range(divisions_y):
        for col_idx in range(divisions_x):
            node_0 = row_idx * stride_y + col_idx
            node_1 = node_0 + 1
            node_2 = (row_idx + 1) * stride_y + (col_idx + 1)
            node_3 = (row_idx + 1) * stride_y + col_idx
            connect[elem_idx] = [node_0, node_1, node_2, node_3]
            elem_idx += 1

    # Planar UV mapping normalized [0, 1] for direct texture comparison
    uvs = np.zeros((total_nodes, 2), dtype=np.float64)
    uvs[:, 0] = (coords[:, 0] + 0.5 * width) / width
    uvs[:, 1] = (0.5 * height - coords[:, 1]) / height

    return coords, connect, uvs


def render_direct_texture_reference(
    mesh_type: int,
    coords: np.ndarray,
    connect: np.ndarray,
    disp: np.ndarray | None,
    uvs: np.ndarray,
    texture: np.ndarray,
    camera: riley.Camera,
) -> np.ndarray:
    """Render directly with Riley's 3D texture shader for ground truth."""
    if texture.ndim == 3 and texture.shape[2] in (1, 3):
        texture_chw = np.transpose(texture, (2, 0, 1))
    elif texture.ndim == 2:
        texture_chw = texture[None, :, :]
    else:
        texture_chw = texture

    mesh_input = riley.Mesh(
        mesh_type=mesh_type,
        coords=coords,
        connect=connect,
        disp=disp,
        shader_type=int(riley.ShaderType.tex_rgb),
        uvs=uvs,
        texture=texture_chw,
        bits=8,
        scaling_type=int(riley.ScaleStrategy.none),
    )

    config = riley.RasterConfig(
        total_threads=1,
        save_strategy=int(riley.SaveStrategy.memory),
        background_value=0.0,
    )

    render_output = riley.raster(
        cameras=[camera],
        meshes=[mesh_input],
        config=config,
    )

    # Convert from [frames, cameras, channels, height, width] to [H, W, C]
    image_channels = render_output[0, 0]
    image_hwc = np.transpose(image_channels, (1, 2, 0))
    return np.clip(image_hwc, 0.0, 255.0).astype(np.uint8)


def calc_image_metrics(
    image_a: np.ndarray,
    image_b: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate MAE, RMSE and max absolute error over valid pixels."""
    if not np.any(valid_mask):
        return 0.0, 0.0, 0.0

    diff = np.abs(
        image_a[valid_mask].astype(np.float64) -
        image_b[valid_mask].astype(np.float64),
    )
    max_err = float(np.max(diff))
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return max_err, mae, rmse


def save_comparison_image(
    name: str,
    warped_img: np.ndarray,
    direct_img: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Save side-by-side comparison: Warped | Direct 3D | Absolute Diff."""
    diff = np.abs(
        warped_img.astype(np.float64) - direct_img.astype(np.float64),
    )
    diff[~valid_mask] = 0.0
    diff_vis = np.clip(diff * 5.0, 0.0, 255.0).astype(np.uint8)

    side_by_side = np.hstack((warped_img, direct_img, diff_vis))
    out_path = OUT_DIR / f"{name}_comparison.png"
    Image.fromarray(side_by_side).save(out_path)
    print(f"  Saved comparison to {out_path}")


def run_all_cases() -> None:
    """Execute full test suite of exaggerated deformation cases."""
    print("=" * 80)
    print("Running Projected Image Warping Verification Cases")
    print("=" * 80)

    # 1. Setup Base Geometry and Texture
    coords_ref, connect, uvs_ref = build_flat_plate_mesh(
        width=100.0,
        height=100.0,
        divisions_x=20,
        divisions_y=20,
    )
    mesh_type = int(riley.MeshType.quad4newton)
    node_num = coords_ref.shape[0]

    # Load high-resolution speckle texture
    speckle_path = Path("texture/speckle_rgb.bmp")
    if not speckle_path.is_file():
        speckle_path = Path("texture/speckle.bmp")

    raw_texture = Image.open(speckle_path).convert("RGB")
    texture_np = np.array(raw_texture, dtype=np.uint8)

    # Reference camera looking straight at the plate
    camera_pixels = (800, 800)
    pixel_size = (1.0e-4, 1.0e-4)
    focal_length = 0.100  # 100mm
    rot_world_ref = (0.0, 0.0, 0.0)

    roi_cent = riley.roi_cent_from_coords(coords_ref)
    pos_world_ref = riley.pos_fill_frame_from_rot(
        coords_ref,
        camera_pixels,
        pixel_size,
        focal_length,
        rot_world_ref,
        1.05,
    )

    camera_ref = riley.Camera(
        pixels_num=camera_pixels,
        pixels_size=pixel_size,
        pos_world=pos_world_ref,
        rot_world=rot_world_ref,
        roi_cent_world=roi_cent,
        focal_length=focal_length,
        sub_sample=2,
    )

    # Render Reference Image from Reference Camera
    image_ref = render_direct_texture_reference(
        mesh_type=mesh_type,
        coords=coords_ref,
        connect=connect,
        disp=None,
        uvs=uvs_ref,
        texture=texture_np,
        camera=camera_ref,
    )
    Image.fromarray(image_ref).save(OUT_DIR / "00_reference_image.png")
    mean_val = image_ref.mean()
    print(f"Rendered base reference image (mean intensity={mean_val:.1f}).")

    # --------------------------------------------------------------------------
    # Case 1: Identity / Undeformed
    # --------------------------------------------------------------------------
    print("\nCase 1: Identity (Zero Deformation)...")
    result_1 = execute_projected_warp(
        mesh_type=mesh_type,
        coords_undeformed=coords_ref,
        connect=connect,
        disp=None,
        camera_target=camera_ref,
        camera_ref=camera_ref,
        image_ref=image_ref,
    )
    direct_1 = render_direct_texture_reference(
        mesh_type=mesh_type,
        coords=coords_ref,
        connect=connect,
        disp=None,
        uvs=uvs_ref,
        texture=texture_np,
        camera=camera_ref,
    )
    max_err, mae, rmse = calc_image_metrics(
        result_1.warped_image,
        direct_1,
        result_1.valid_mask,
    )
    print(
        f"  Identity metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, Max={max_err:.3f}"
    )
    save_comparison_image(
        "case1_identity",
        result_1.warped_image,
        direct_1,
        result_1.valid_mask,
    )
    assert rmse < 3.0, f"Identity warp error too large: RMSE={rmse}"

    # --------------------------------------------------------------------------
    # Case 2: Pure In-Plane Translation (Exaggerated: +10mm X, -5mm Y)
    # --------------------------------------------------------------------------
    print("\nCase 2: In-Plane Rigid Translation (+10mm X, -5mm Y)...")
    disp_2 = np.zeros((1, node_num, 3), dtype=np.float64)
    disp_2[0, :, 0] = 10.0
    disp_2[0, :, 1] = -5.0

    result_2 = execute_projected_warp(
        mesh_type=mesh_type,
        coords_undeformed=coords_ref,
        connect=connect,
        disp=disp_2,
        camera_target=camera_ref,
        camera_ref=camera_ref,
        image_ref=image_ref,
    )
    direct_2 = render_direct_texture_reference(
        mesh_type=mesh_type,
        coords=coords_ref,
        connect=connect,
        disp=disp_2,
        uvs=uvs_ref,
        texture=texture_np,
        camera=camera_ref,
    )
    max_err, mae, rmse = calc_image_metrics(
        result_2.warped_image,
        direct_2,
        result_2.valid_mask,
    )
    print(
        f"  Translation metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, "
        f"Max={max_err:.3f}"
    )
    save_comparison_image(
        "case2_translation",
        result_2.warped_image,
        direct_2,
        result_2.valid_mask,
    )
    assert rmse < 10.0, f"Translation warp error too large: RMSE={rmse}"

    # --------------------------------------------------------------------------
    # Case 3: Pure In-Plane Rotation (Exaggerated: 30 degrees)
    # --------------------------------------------------------------------------
    print("\nCase 3: In-Plane Rotation (30 degrees)...")
    theta = np.deg2rad(30.0)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    coords_rot = coords_ref @ rot_z.T
    disp_3 = np.zeros((1, node_num, 3), dtype=np.float64)
    disp_3[0] = coords_rot - coords_ref

    result_3 = execute_projected_warp(
        mesh_type=mesh_type,
        coords_undeformed=coords_ref,
        connect=connect,
        disp=disp_3,
        camera_target=camera_ref,
        camera_ref=camera_ref,
        image_ref=image_ref,
    )
    direct_3 = render_direct_texture_reference(
        mesh_type=mesh_type,
        coords=coords_ref,
        connect=connect,
        disp=disp_3,
        uvs=uvs_ref,
        texture=texture_np,
        camera=camera_ref,
    )
    max_err, mae, rmse = calc_image_metrics(
        result_3.warped_image,
        direct_3,
        result_3.valid_mask,
    )
    print(
        f"  Rotation 30 deg metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, "
        f"Max={max_err:.3f}"
    )
    save_comparison_image(
        "case3_rotation_30deg",
        result_3.warped_image,
        direct_3,
        result_3.valid_mask,
    )
    assert rmse < 10.0, f"Rotation warp error too large: RMSE={rmse}"

    # --------------------------------------------------------------------------
    # Case 4: Out-of-Plane Plunge / Translation (+25mm Z Zoom)
    # --------------------------------------------------------------------------
    print("\nCase 4: Out-of-Plane Plunge (+25mm Z Towards Camera)...")
    disp_4 = np.zeros((1, node_num, 3), dtype=np.float64)
    disp_4[0, :, 2] = 25.0

    result_4 = execute_projected_warp(
        mesh_type=mesh_type,
        coords_undeformed=coords_ref,
        connect=connect,
        disp=disp_4,
        camera_target=camera_ref,
        camera_ref=camera_ref,
        image_ref=image_ref,
    )
    direct_4 = render_direct_texture_reference(
        mesh_type=mesh_type,
        coords=coords_ref,
        connect=connect,
        disp=disp_4,
        uvs=uvs_ref,
        texture=texture_np,
        camera=camera_ref,
    )
    max_err, mae, rmse = calc_image_metrics(
        result_4.warped_image,
        direct_4,
        result_4.valid_mask,
    )
    print(
        f"  Plunge metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, Max={max_err:.3f}"
    )
    save_comparison_image(
        "case4_plunge",
        result_4.warped_image,
        direct_4,
        result_4.valid_mask,
    )
    assert rmse < 15.0, f"Plunge warp error too large: RMSE={rmse}"

    # --------------------------------------------------------------------------
    # Case 5: Out-of-Plane 3D Tilt (15 degrees around Y axis)
    # --------------------------------------------------------------------------
    print("\nCase 5: Out-of-Plane Tilt (15 degrees around Y axis)...")
    phi = np.deg2rad(15.0)
    rot_y = np.array([
        [np.cos(phi), 0.0, np.sin(phi)],
        [0.0, 1.0, 0.0],
        [-np.sin(phi), 0.0, np.cos(phi)],
    ])
    coords_tilt = coords_ref @ rot_y.T
    disp_5 = np.zeros((1, node_num, 3), dtype=np.float64)
    disp_5[0] = coords_tilt - coords_ref

    result_5 = execute_projected_warp(
        mesh_type=mesh_type,
        coords_undeformed=coords_ref,
        connect=connect,
        disp=disp_5,
        camera_target=camera_ref,
        camera_ref=camera_ref,
        image_ref=image_ref,
    )
    direct_5 = render_direct_texture_reference(
        mesh_type=mesh_type,
        coords=coords_ref,
        connect=connect,
        disp=disp_5,
        uvs=uvs_ref,
        texture=texture_np,
        camera=camera_ref,
    )
    max_err, mae, rmse = calc_image_metrics(
        result_5.warped_image,
        direct_5,
        result_5.valid_mask,
    )
    print(
        f"  Tilt metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, Max={max_err:.3f}"
    )
    save_comparison_image(
        "case5_tilt",
        result_5.warped_image,
        direct_5,
        result_5.valid_mask,
    )
    assert rmse < 15.0, f"Tilt warp error too large: RMSE={rmse}"

    # --------------------------------------------------------------------------
    # Case 6: Stereo Camera View (Target Camera Rotated by 15 degrees)
    # --------------------------------------------------------------------------
    print("\nCase 6: Stereo-DIC Target Camera View (15 deg angle)...")
    stereo_angle = np.deg2rad(15.0)
    cam_dist = float(pos_world_ref[2])

    cam_pos_stereo = (
        cam_dist * np.sin(stereo_angle),
        0.0,
        cam_dist * np.cos(stereo_angle),
    )
    # Camera rotates around Y axis to point at (0, 0, 0)
    cam_rot_stereo = (0.0, stereo_angle, 0.0)

    camera_stereo = riley.Camera(
        pixels_num=camera_pixels,
        pixels_size=pixel_size,
        pos_world=cam_pos_stereo,
        rot_world=cam_rot_stereo,
        roi_cent_world=(0.0, 0.0, 0.0),
        focal_length=focal_length,
        sub_sample=2,
    )

    # Complex sinusoidal + tension deformation
    disp_6 = np.zeros((1, node_num, 3), dtype=np.float64)
    disp_6[0, :, 0] = 0.02 * coords_ref[:, 0]  # 2% tension
    disp_6[0, :, 1] = 3.0 * np.sin(2.0 * np.pi * coords_ref[:, 0] / 100.0)

    result_6 = execute_projected_warp(
        mesh_type=mesh_type,
        coords_undeformed=coords_ref,
        connect=connect,
        disp=disp_6,
        camera_target=camera_stereo,
        camera_ref=camera_ref,
        image_ref=image_ref,
    )
    direct_6 = render_direct_texture_reference(
        mesh_type=mesh_type,
        coords=coords_ref,
        connect=connect,
        disp=disp_6,
        uvs=uvs_ref,
        texture=texture_np,
        camera=camera_stereo,
    )
    max_err, mae, rmse = calc_image_metrics(
        result_6.warped_image,
        direct_6,
        result_6.valid_mask,
    )
    print(
        f"  Stereo-DIC metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, "
        f"Max={max_err:.3f}"
    )
    save_comparison_image(
        "case6_stereo_complex",
        result_6.warped_image,
        direct_6,
        result_6.valid_mask,
    )
    assert rmse < 15.0, f"Stereo-DIC warp error too large: RMSE={rmse}"

    print("\n" + "=" * 80)
    print("All verification cases passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_cases()
