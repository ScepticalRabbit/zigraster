# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Unit tests for projected image warping prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import riley

# Add prototype directory to path
PROTOTYPE_DIR = Path(__file__).parents[3] / "prototype" / "proj-image-warp"
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

from warp_engine import (
    calc_rot_matrix_zyx,
    execute_projected_warp,
    project_world_to_cam_pixels,
    render_projected_coord_map,
    warp_reference_image,
)


def test_rotation_matrix_identity() -> None:
    """Test ZYX Euler rotation matrix at zero angles equals identity."""
    rot_mat = calc_rot_matrix_zyx(0.0, 0.0, 0.0)
    identity_mat = np.eye(3, dtype=np.float64)
    np.testing.assert_allclose(rot_mat, identity_mat, atol=1.0e-12)


def test_project_world_to_cam_pixels_center() -> None:
    """Test pinhole projection of origin point to image center."""
    camera = riley.Camera(
        pixels_num=(640, 480),
        pixels_size=(1.0e-5, 1.0e-5),
        pos_world=(0.0, 0.0, 1.0),
        rot_world=(0.0, 0.0, 0.0),
        roi_cent_world=(0.0, 0.0, 0.0),
        focal_length=0.050,
        sub_sample=1,
    )
    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    pixel_coords = project_world_to_cam_pixels(coords, camera)

    # Center pixel in 640x480 is (320, 240)
    expected = np.array([[320.0, 240.0]], dtype=np.float64)
    np.testing.assert_allclose(pixel_coords, expected, atol=1.0e-10)


def test_identity_warp_exact() -> None:
    """Test identity warp matches base reference image exactly."""
    coords = np.array([
        [-10.0, -10.0, 0.0],
        [ 10.0, -10.0, 0.0],
        [ 10.0,  10.0, 0.0],
        [-10.0,  10.0, 0.0],
    ], dtype=np.float64)
    connect = np.array([[0, 1, 2, 3]], dtype=np.uint32)

    pixels_num = (50, 50)
    pixels_size = (1.0e-3, 1.0e-3)
    focal_length = 0.100
    rot_world = (0.0, 0.0, 0.0)
    roi_cent = (0.0, 0.0, 0.0)
    pos_world = (0.0, 0.0, 2.0)

    camera = riley.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=pos_world,
        rot_world=rot_world,
        roi_cent_world=roi_cent,
        focal_length=focal_length,
        sub_sample=1,
    )

    # Generate synthetic checkerboard reference image
    ref_image = np.zeros((50, 50), dtype=np.float64)
    ref_image[10:40, 10:40] = 200.0

    result = execute_projected_warp(
        mesh_type=int(riley.MeshType.quad4newton),
        coords_undeformed=coords,
        connect=connect,
        disp=None,
        camera_target=camera,
        camera_ref=camera,
        image_ref=ref_image,
        interp_order=1,
    )

    valid_mask = result.valid_mask
    assert np.any(valid_mask)
    diff = np.abs(result.warped_image[valid_mask] - ref_image[valid_mask])
    assert np.max(diff) < 1.0e-6
