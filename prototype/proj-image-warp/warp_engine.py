# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Projected image warping engine using Riley nodal field rasterisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.ndimage
from PIL import Image

import riley


@dataclass(slots=True)
class ProjectedWarpResult:
    """Result of a projected image warping operation."""
    warped_image: np.ndarray
    coord_map: np.ndarray
    valid_mask: np.ndarray


def calc_rot_matrix_zyx(
    alpha_z: float,
    beta_y: float,
    gamma_x: float,
) -> np.ndarray:
    """Calculate the 3x3 rotation matrix for intrinsic ZYX Euler angles."""
    cos_a = np.cos(alpha_z)
    sin_a = np.sin(alpha_z)
    cos_b = np.cos(beta_y)
    sin_b = np.sin(beta_y)
    cos_g = np.cos(gamma_x)
    sin_g = np.sin(gamma_x)

    rot_mat = np.zeros((3, 3), dtype=np.float64)

    rot_mat[0, 0] = cos_a * cos_b
    rot_mat[0, 1] = cos_a * sin_b * sin_g - sin_a * cos_g
    rot_mat[0, 2] = cos_a * sin_b * cos_g + sin_a * sin_g

    rot_mat[1, 0] = sin_a * cos_b
    rot_mat[1, 1] = sin_a * sin_b * sin_g + cos_a * cos_g
    rot_mat[1, 2] = sin_a * sin_b * cos_g - cos_a * sin_g

    rot_mat[2, 0] = -sin_b
    rot_mat[2, 1] = cos_b * sin_g
    rot_mat[2, 2] = cos_b * cos_g

    return rot_mat


def project_world_to_cam_pixels(
    coords_world: np.ndarray,
    camera: riley.Camera,
) -> np.ndarray:
    """Project Nx3 world coordinates to Nx2 camera sensor pixel coords."""
    node_num = coords_world.shape[0]
    pos_world = np.array(camera.pos_world, dtype=np.float64)
    rot_mat = calc_rot_matrix_zyx(
        camera.rot_world[0],
        camera.rot_world[1],
        camera.rot_world[2],
    )

    # World to camera coordinates: X_cam = R^T * (X_world - pos_world)
    diff_world = coords_world - pos_world
    coords_cam = diff_world @ rot_mat

    pixels_num = np.array(camera.pixels_num, dtype=np.float64)
    pixels_size = np.array(camera.pixels_size, dtype=np.float64)
    focal_length = float(camera.focal_length)

    focal_px_x = focal_length / pixels_size[0]
    focal_px_y = focal_length / pixels_size[1]

    principal_point_x = 0.5 * pixels_num[0]
    principal_point_y = 0.5 * pixels_num[1]

    # Camera looks down -Z axis in OpenGL convention
    inv_neg_z = 1.0 / (-coords_cam[:, 2])

    coords_norm_x = coords_cam[:, 0] * inv_neg_z
    coords_norm_y = coords_cam[:, 1] * inv_neg_z

    pixel_coords = np.zeros((node_num, 2), dtype=np.float64)
    pixel_coords[:, 0] = focal_px_x * coords_norm_x + principal_point_x
    pixel_coords[:, 1] = principal_point_y - focal_px_y * coords_norm_y

    return pixel_coords


def build_nodal_ref_coord_field(
    coords_undeformed: np.ndarray,
    camera_ref: riley.Camera,
) -> np.ndarray:
    """Compute projected 2D reference coordinates as a nodal field.

    Returns an array of shape [1, total_nodes, 2] suitable for Riley nodal
    shader input.
    """
    ref_pixel_coords = project_world_to_cam_pixels(
        coords_undeformed,
        camera_ref,
    )
    node_num = coords_undeformed.shape[0]
    field_data = np.zeros((1, node_num, 2), dtype=np.float64)
    field_data[0, :, 0] = ref_pixel_coords[:, 0]
    field_data[0, :, 1] = ref_pixel_coords[:, 1]
    return field_data


def render_projected_coord_map(
    mesh_type: int,
    coords_undeformed: np.ndarray,
    connect: np.ndarray,
    disp: np.ndarray | None,
    camera_target: riley.Camera,
    camera_ref: riley.Camera,
    config: riley.RasterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render dense reference coordinate mapping using Riley nodal shader.

    Returns:
        coord_map: ndarray of shape [height, width, 2] with (x_ref, y_ref).
        valid_mask: boolean ndarray of shape [height, width] indicating
                    covered pixels.
    """
    nodal_field = build_nodal_ref_coord_field(coords_undeformed, camera_ref)

    mesh_input = riley.Mesh(
        mesh_type=mesh_type,
        coords=coords_undeformed,
        connect=connect,
        disp=disp,
        shader_type=int(riley.ShaderType.nodal),
        nodal_field=nodal_field,
        bits=0,
        scaling_type=int(riley.ScaleStrategy.none),
    )

    if config is None:
        config = riley.RasterConfig(
            total_threads=1,
            save_strategy=int(riley.SaveStrategy.memory),
            image_save_mode=int(riley.ImageSaveMode.multifield),
            background_value=-1.0e9,
        )

    # Render through Riley Cython bindings
    render_output = riley.raster(
        cameras=[camera_target],
        meshes=[mesh_input],
        config=config,
    )

    # render_output shape: [frames, cameras, channels, height, width]
    # Channels 0 and 1 correspond to x_ref and y_ref
    rendered_field = render_output[0, 0]
    channel_x = rendered_field[0]
    channel_y = rendered_field[1]

    # Valid mask where pixel was covered by rasteriser (not background)
    valid_mask = channel_x > -1.0e8

    height, width = channel_x.shape
    coord_map = np.zeros((height, width, 2), dtype=np.float64)
    coord_map[:, :, 0] = channel_x
    coord_map[:, :, 1] = channel_y

    return coord_map, valid_mask


def warp_reference_image(
    image_ref: np.ndarray,
    coord_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    interp_order: int = 3,
    background_value: float = 0.0,
) -> np.ndarray:
    """Warp a 2D reference image using dense coordinates (x_ref, y_ref).

    Continuous screen pixel centers (col + 0.5, row + 0.5) are converted to
    array sample coordinates by subtracting 0.5.

    Args:
        image_ref: 2D or 3D numpy array [height, width] or [height, width, C].
        coord_map: Dense mapping [out_height, out_width, 2] with (x, y).
        valid_mask: Optional boolean mask of covered pixels.
        interp_order: Spline interpolation order (1=linear, 3=cubic).
        background_value: Fill value for unmapped pixels.

    Returns:
        Warped image array with matching dtype and channel shape.
    """
    out_height, out_width = coord_map.shape[:2]

    # Continuous screen coordinate center -> array index offset
    sample_y = (coord_map[:, :, 1] - 0.5).ravel()
    sample_x = (coord_map[:, :, 0] - 0.5).ravel()
    coordinates = np.vstack((sample_y, sample_x))

    if image_ref.ndim == 2:
        warped_flat = scipy.ndimage.map_coordinates(
            image_ref,
            coordinates,
            order=interp_order,
            mode="nearest",
        )
        warped_image = warped_flat.reshape((out_height, out_width))
        if valid_mask is not None:
            warped_image[~valid_mask] = background_value

    elif image_ref.ndim == 3:
        num_channels = image_ref.shape[2]
        warped_image = np.zeros(
            (out_height, out_width, num_channels),
            dtype=image_ref.dtype,
        )
        for channel_idx in range(num_channels):
            channel_data = image_ref[:, :, channel_idx]
            warped_channel_flat = scipy.ndimage.map_coordinates(
                channel_data,
                coordinates,
                order=interp_order,
                mode="nearest",
            )
            warped_channel = warped_channel_flat.reshape(
                (out_height, out_width),
            )
            if valid_mask is not None:
                warped_channel[~valid_mask] = background_value
            warped_image[:, :, channel_idx] = warped_channel

    else:
        raise ValueError(
            f"Unsupported reference image dimensions: {image_ref.ndim}",
        )

    return warped_image


def execute_projected_warp(
    mesh_type: int,
    coords_undeformed: np.ndarray,
    connect: np.ndarray,
    disp: np.ndarray | None,
    camera_target: riley.Camera,
    camera_ref: riley.Camera,
    image_ref: np.ndarray,
    interp_order: int = 3,
    background_value: float = 0.0,
) -> ProjectedWarpResult:
    """Execute complete projected image warping pipeline."""
    coord_map, valid_mask = render_projected_coord_map(
        mesh_type=mesh_type,
        coords_undeformed=coords_undeformed,
        connect=connect,
        disp=disp,
        camera_target=camera_target,
        camera_ref=camera_ref,
    )

    warped_image = warp_reference_image(
        image_ref=image_ref,
        coord_map=coord_map,
        valid_mask=valid_mask,
        interp_order=interp_order,
        background_value=background_value,
    )

    return ProjectedWarpResult(
        warped_image=warped_image,
        coord_map=coord_map,
        valid_mask=valid_mask,
    )
