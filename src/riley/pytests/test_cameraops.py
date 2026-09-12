"""Tests for camera framing and positioning operations."""

import math
from pathlib import Path
import numpy as np
import pytest

import riley


def test_extended_distortion_camera_io_roundtrip(tmp_path: Path) -> None:
    camera = riley.Camera(
        pixels_num=(100, 80),
        pixels_size=(1.0e-5, 1.1e-5),
        pos_world=(0.0, 0.0, 0.2),
        rot_world=(0.0, 0.0, 0.0),
        roi_cent_world=(0.0, 0.0, 0.0),
        focal_length=0.05,
        sub_sample=1,
        distortion_model=2,
        distortion_s1=1.1e-3,
        distortion_s2=-1.2e-3,
        distortion_s3=1.3e-3,
        distortion_s4=-1.4e-3,
        distortion_tau_x=0.021,
        distortion_tau_y=-0.034,
    )
    riley.save_camera(str(tmp_path), "camera.csv", 0, camera)
    loaded = riley.load_camera(str(tmp_path), "camera.csv")

    assert loaded.distortion_s1 == pytest.approx(camera.distortion_s1)
    assert loaded.distortion_s2 == pytest.approx(camera.distortion_s2)
    assert loaded.distortion_s3 == pytest.approx(camera.distortion_s3)
    assert loaded.distortion_s4 == pytest.approx(camera.distortion_s4)
    assert loaded.distortion_tau_x == pytest.approx(camera.distortion_tau_x)
    assert loaded.distortion_tau_y == pytest.approx(camera.distortion_tau_y)


def test_coverage_and_fov_scale_roundtrip() -> None:
    coverage = 0.8
    fov_scale = riley.coverage_to_fov_scale(coverage)
    assert fov_scale == pytest.approx(1.25)
    roundtrip = riley.fov_scale_to_coverage(fov_scale)
    assert roundtrip == pytest.approx(0.8)


def test_pos_frame_coords_fit_modes() -> None:
    coords = np.array(
        [
            [-10.0, -5.0, 0.0],
            [10.0, -5.0, 0.0],
            [10.0, 5.0, 0.0],
            [-10.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    pixels_num = (100, 100)
    pixels_size = (0.1, 0.1)
    focal_length = 10.0
    rot = (0.0, 0.0, 0.0)

    # contain mode should fit the wider dimension (X=20 => dist=20)
    pos_contain = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot,
        fov_scale=1.0,
        fit_mode=riley.FrameFitMode.contain,
    )
    np.testing.assert_allclose(pos_contain, (0.0, 0.0, 20.0), atol=1e-5)

    # cover mode should fit the smaller dimension (Y=10 => dist=10)
    pos_cover = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot,
        fov_scale=1.0,
        fit_mode="cover",
    )
    np.testing.assert_allclose(pos_cover, (0.0, 0.0, 10.0), atol=1e-5)

    # horizontal mode should fit X=20 => dist=20
    pos_horiz = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot,
        fov_scale=1.0,
        fit_mode="horizontal",
    )
    np.testing.assert_allclose(pos_horiz, (0.0, 0.0, 20.0), atol=1e-5)

    # vertical mode should fit Y=10 => dist=10
    pos_vert = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot,
        fov_scale=1.0,
        fit_mode="vertical",
    )
    np.testing.assert_allclose(pos_vert, (0.0, 0.0, 10.0), atol=1e-5)


def test_pos_frame_coords_with_target() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    target = (10.0, 0.0, 0.0)
    pos = riley.pos_frame_coords(
        coords,
        (100, 100),
        (0.1, 0.1),
        10.0,
        (0.0, 0.0, 0.0),
        fov_scale=1.0,
        target=target,
    )
    np.testing.assert_allclose(pos, (10.0, 0.0, 20.0), atol=1e-5)


def test_pos_orbit_cam_placement() -> None:
    target = (0.0, 0.0, 0.0)
    distance = 100.0
    pos, rot = riley.pos_orbit_cam(target, 0.0, 0.0, distance)
    np.testing.assert_allclose(pos, (100.0, 0.0, 0.0), atol=1e-5)


def test_pos_stereo_pair_symmetry() -> None:
    target = (0.0, 0.0, 0.0)
    distance = 100.0
    stereo_angle = math.pi / 6.0
    cam0_pos, cam0_rot, cam1_pos, cam1_rot = riley.pos_stereo_pair(
        target,
        distance,
        stereo_angle,
        0.0,
    )
    diff = np.array(cam0_pos) - np.array(cam1_pos)
    baseline = np.linalg.norm(diff)
    expected_baseline = 2.0 * distance * math.sin(0.5 * stereo_angle)
    assert baseline == pytest.approx(expected_baseline, rel=1e-5)


def test_calc_pixel_resolution() -> None:
    cam = riley.Camera(
        pixels_num=(100, 100),
        pixels_size=(0.1, 0.1),
        pos_world=(0.0, 0.0, 20.0),
        rot_world=(0.0, 0.0, 0.0),
        roi_cent_world=(0.0, 0.0, 0.0),
        focal_length=10.0,
        sub_sample=1,
    )
    res = riley.calc_pixel_resolution(cam, (0.0, 0.0, 0.0))
    # Distance is 20, focal is 10 => magnification is 10/20 = 0.5.
    # Pixel size is 0.1 => mm per pixel = 0.1 / 0.5 = 0.2.
    assert res == pytest.approx(0.2, rel=1e-5)


def test_pos_fill_frame_from_rot_deprecated_wrapper() -> None:
    coords = np.array(
        [
            [-10.0, -5.0, 0.0],
            [10.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.deprecated_call():
        pos = riley.pos_fill_frame_from_rot(
            coords,
            (100, 100),
            (0.1, 0.1),
            10.0,
            (0.0, 0.0, 0.0),
            frame_fill=1.0,
        )
    np.testing.assert_allclose(pos, (0.0, 0.0, 20.0), atol=1e-5)
