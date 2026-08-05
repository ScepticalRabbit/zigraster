# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
import cython
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.riley.cyth import riley as cr


@dataclass(slots=True)
class Camera:
    pixels_num: tuple[int, int]
    pixels_size: tuple[float, float]
    pos_world: tuple[float, float, float]
    rot_world: tuple[float, float, float]
    roi_cent_world: tuple[float, float, float]
    focal_length: float
    sub_sample: int
    distortion_model: int = 0
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    distortion_k3: float = 0.0
    distortion_k4: float = 0.0
    distortion_k5: float = 0.0
    distortion_k6: float = 0.0
    distortion_p1: float = 0.0
    distortion_p2: float = 0.0
    distortion_poly_order: int = 2
    distortion_poly_has_forward: bool = False
    distortion_poly_has_inverse: bool = False
    distortion_poly_forward_u: tuple[float, float, float, float, float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    distortion_poly_forward_v: tuple[float, float, float, float, float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    distortion_poly_inverse_u: tuple[float, float, float, float, float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    distortion_poly_inverse_v: tuple[float, float, float, float, float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    coord_sys: int = 0
    subpixel_center_map: int = 1
    psf_type: int = 0
    psf_sigma_x: float = 0.0
    psf_sigma_y: float = 0.0
    psf_theta: float = 0.0
    psf_support_rad: float = 0.0
    psf_separable: int = 1


CameraInput = Camera


@dataclass(slots=True)
class FuncShaderParams:
    coord_scale: tuple[float, float] = (1.0, 1.0)
    coord_offset: tuple[float, float] = (0.0, 0.0)
    output_scale: float = 1.0
    output_offset: float = 0.0
    constant_value: float = 0.5
    constant_value_rgb: tuple[float, float, float] = (0.2, 0.5, 0.8)
    linear_coeffs: tuple[float, float, float] = (0.5, 0.25, 0.2)
    linear_coeffs_rgb: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (0.5, 0.25, 0.0),
        (0.5, 0.0, 0.25),
        (0.5, 0.15, -0.15),
    )
    quadratic_coeffs: tuple[float, float, float, float, float, float] = (
        0.35, 0.2, 0.15, 0.1, -0.08, 0.06,
    )
    quadratic_coeffs_rgb: tuple[
        tuple[float, float, float, float, float, float],
        tuple[float, float, float, float, float, float],
        tuple[float, float, float, float, float, float],
    ] = (
        (0.3, 0.0, 0.0, 0.2, 0.0, 0.0),
        (0.3, 0.0, 0.0, 0.0, 0.0, 0.2),
        (0.3, 0.0, 0.0, 0.0, 0.12, 0.0),
    )
    wave_num_scalar: tuple[float, float] = (6.0, 5.0)
    wave_num_rgb: tuple[float, float, float] = (6.0, 6.0, 4.0)
    sinusoidal_bias: float = 0.5
    sinusoidal_amplitudes: tuple[float, float] = (0.25, 0.2)
    sinusoidal_bias_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    sinusoidal_amplitudes_rgb: tuple[float, float, float] = (
        0.25, 0.25, 0.2,
    )
    checker_levels: tuple[float, float] = (0.0, 1.0)
    checker_smooth_frequency: float = 8.0
    lambertian_coeffs: tuple[float, float] = (0.5, 0.5)
    lambertian_coeffs_rgb: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = (
        (0.5, 0.5),
        (0.375, 0.375),
        (0.25, 0.25),
    )
    eggbox_mean: float = 0.5
    eggbox_contrast: float = 0.4
    eggbox_pitch: tuple[float, float] = (1.0, 1.0)
    eggbox_phase: tuple[float, float] = (0.0, 0.0)
    extra: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )


@dataclass(slots=True)
class Mesh:
    mesh_type: int
    coords: np.ndarray
    connect: np.ndarray
    disp: np.ndarray | None = None
    shader_type: int = 0
    uvs: np.ndarray | None = None
    texture: np.ndarray | None = None
    texture_storage: int = 0
    sample: int = 2
    sample_mode: int = 2
    bits: int = 8
    scaling_type: int = 0
    scaling_min: float = 0.0
    scaling_max: float = 0.0
    nodal_field: np.ndarray | None = None
    scale_over: int = 1
    func_shader_builtin: int = 0
    func_shader_coord_mode: int = 1
    func_shader_params: FuncShaderParams = field(
        default_factory=FuncShaderParams,
    )
    normal_type: int = 0


MeshInput = Mesh


@dataclass(slots=True)
class RasterConfig:
    render_mode: int = 0
    total_threads: int = 1
    frame_batch_size_per_group: int = 1
    max_geom_jobs_in_flight_per_group: int = 1
    max_geom_workers_per_job: int = 1
    geom_scheduling_mode: int = 2
    max_raster_workers_per_job: int = 1
    save_strategy: int = 1
    image_save_mode: int = 2
    hull_mode: int = 1
    newton_seed_mode: int = 0
    newton_seed_reuse: int = 0
    report: int = 1
    tile_size_min: int = 1
    tile_size_max: int = 256
    background_value: float = 0.0
    disk_save_overlap: bool = False
    tile_size_override: int = 0
    save_frame_buffer_count: int = 3
    save_format: int = 3
    save_bits: int = 8
    save_scaling: int = 0
    save_scaling_min: float = 0.0
    save_scaling_max: float = 0.0
    full_stats_save_solver_csv: bool = False
    full_stats_save_iter_map: bool = True
    full_stats_save_xi_map: bool = True
    full_stats_save_eta_map: bool = True
    full_stats_save_conv_map: bool = True
    full_stats_save_jac_det_map: bool = True
    full_stats_save_tile_timing_map: bool = True
    full_stats_save_tile_density_map: bool = True
    full_stats_save_tile_occupancy_map: bool = True
    full_stats_save_depth_map: bool = True
    full_stats_save_earlyout_map: bool = True
    full_stats_save_pixel_occupancy_map: bool = True
    full_stats_save_normals_map: bool = False


class MeshType(IntEnum):
    tri3 = 0
    tri3opt = 1
    tri6 = 2
    quad4ibi = 3
    quad4newton = 4
    quad8 = 5
    quad9 = 6


class ShaderType(IntEnum):
    tex = 0
    tex_rgb = 1
    nodal = 2
    func = 3
    func_rgb = 4
    nodal_rgb = 5


class TextureStorage(IntEnum):
    u8 = 0
    u16 = 1
    floating = 2


class RenderMode(IntEnum):
    in_order = 0
    offline = 1


class GeometrySchedulingMode(IntEnum):
    spread = 0
    pack = 1
    auto = 2


class SaveStrategy(IntEnum):
    disk = 0
    memory = 1
    both = 2
    none = 3


class ImageSaveMode(IntEnum):
    grey = 0
    rgb = 1
    multifield = 2


class ReportMode(IntEnum):
    off = 0
    bench = 1
    full_stats = 2


class SubPixelCenterMap(IntEnum):
    full_in_mem = 0
    per_tile = 1
    affine_jac = 2


class TextureSample(IntEnum):
    nearest = 0
    linear = 1
    cubic_catmull_rom = 2
    cubic_mitchell_netravali = 3
    lanczos3 = 4
    cubic_bspline = 5
    quintic_bspline = 6
    lanczos2 = 7


class TextureSampleMode(IntEnum):
    direct = 0
    lut = 1
    lut_lerp = 2


class ScaleStrategy(IntEnum):
    none = 0
    auto = 1
    fixed = 2
    frac = 3


class ScaleOver(IntEnum):
    within_frames = 0
    over_frames = 1


class FuncShaderBuiltin(IntEnum):
    constant = 0
    linear = 1
    quadratic = 2
    sinusoidal = 3
    sinusoidal_approx = 4
    checker = 5
    checker_smooth = 6
    lambertian_normal_z = 7
    eggbox = 8


class FuncCoordMode(IntEnum):
    uv = 0
    parametric = 1
    world_reference = 2
    world_deformed = 3


class NormalType(IntEnum):
    none = 0
    exact = 1
    averaged = 2


class HullMode(IntEnum):
    off = 0
    on_no_fallback = 1
    on_convex_fallback = 2


class NewtonSeedMode(IntEnum):
    centroid = 0
    hull = 1


class NewtonSeedReuse(IntEnum):
    off = 0
    last_converged = 1


class CameraCoordSys(IntEnum):
    opengl = 0
    opencv = 1


class PsfType(IntEnum):
    pixel_box = 0
    gaussian = 1
    anisotropic_gaussian = 2


class ImageFormat(IntEnum):
    csv = 0
    fimg = 1
    ppm = 2
    bmp = 3
    tiff = 4


@cython.cfunc
def _make_cvec3(vec_in: tuple[float, float, float]) -> cr.CVec3F64:
    return cr.CVec3F64(
        float(vec_in[0]),
        float(vec_in[1]),
        float(vec_in[2]),
    )


@cython.cfunc
def _make_cvec2_f64(vec_in: tuple[float, float]) -> cr.CVec2F64:
    return cr.CVec2F64(float(vec_in[0]), float(vec_in[1]))


@cython.cfunc
def _make_cvec2_u32(vec_in: tuple[int, int]) -> cr.CVec2U32:
    return cr.CVec2U32(int(vec_in[0]), int(vec_in[1]))


@cython.cfunc
def _make_camera_input(camera: Any) -> cr.CCameraInput:
    camera_out: cr.CCameraInput
    idx: cython.Py_ssize_t
    camera_out.pixels_num = _make_cvec2_u32(camera.pixels_num)
    camera_out.pixels_size = _make_cvec2_f64(camera.pixels_size)
    camera_out.pos_world = _make_cvec3(camera.pos_world)
    camera_out.rot_world = _make_cvec3(camera.rot_world)
    camera_out.roi_cent_world = _make_cvec3(camera.roi_cent_world)
    camera_out.focal_length = float(camera.focal_length)
    camera_out.sub_sample = int(camera.sub_sample)
    camera_out.distortion_model = int(camera.distortion_model)
    camera_out.distortion_k1 = float(camera.distortion_k1)
    camera_out.distortion_k2 = float(camera.distortion_k2)
    camera_out.distortion_k3 = float(camera.distortion_k3)
    camera_out.distortion_k4 = float(camera.distortion_k4)
    camera_out.distortion_k5 = float(camera.distortion_k5)
    camera_out.distortion_k6 = float(camera.distortion_k6)
    camera_out.distortion_p1 = float(camera.distortion_p1)
    camera_out.distortion_p2 = float(camera.distortion_p2)
    camera_out.distortion_poly_order = int(camera.distortion_poly_order)
    camera_out.distortion_poly_has_forward = int(camera.distortion_poly_has_forward)
    camera_out.distortion_poly_has_inv = int(camera.distortion_poly_has_inverse)
    for idx in range(10):
        camera_out.distortion_poly_forward_u[idx] = float(
            camera.distortion_poly_forward_u[idx]
        )
        camera_out.distortion_poly_forward_v[idx] = float(
            camera.distortion_poly_forward_v[idx]
        )
        camera_out.distortion_poly_inv_u[idx] = float(
            camera.distortion_poly_inverse_u[idx]
        )
        camera_out.distortion_poly_inv_v[idx] = float(
            camera.distortion_poly_inverse_v[idx]
        )
    camera_out.coord_sys = int(camera.coord_sys)
    camera_out.subpixel_center_map = int(camera.subpixel_center_map)
    camera_out.psf_type = int(camera.psf_type)
    camera_out.psf_sigma_x = float(camera.psf_sigma_x)
    camera_out.psf_sigma_y = float(camera.psf_sigma_y)
    camera_out.psf_theta = float(camera.psf_theta)
    camera_out.psf_supp_rad = float(camera.psf_support_rad)
    camera_out.psf_separable = int(camera.psf_separable)
    return camera_out


def _camera_input_from_c(camera_in: cr.CCameraInput) -> Camera:
    idx: cython.Py_ssize_t
    forward_u = [0.0] * 10
    forward_v = [0.0] * 10
    inverse_u = [0.0] * 10
    inverse_v = [0.0] * 10
    for idx in range(10):
        forward_u[idx] = camera_in.distortion_poly_forward_u[idx]
        forward_v[idx] = camera_in.distortion_poly_forward_v[idx]
        inverse_u[idx] = camera_in.distortion_poly_inv_u[idx]
        inverse_v[idx] = camera_in.distortion_poly_inv_v[idx]
    return Camera(
        pixels_num=(camera_in.pixels_num.x, camera_in.pixels_num.y),
        pixels_size=(camera_in.pixels_size.x, camera_in.pixels_size.y),
        pos_world=(
            camera_in.pos_world.x,
            camera_in.pos_world.y,
            camera_in.pos_world.z,
        ),
        rot_world=(
            camera_in.rot_world.x,
            camera_in.rot_world.y,
            camera_in.rot_world.z,
        ),
        roi_cent_world=(
            camera_in.roi_cent_world.x,
            camera_in.roi_cent_world.y,
            camera_in.roi_cent_world.z,
        ),
        focal_length=camera_in.focal_length,
        sub_sample=camera_in.sub_sample,
        distortion_model=camera_in.distortion_model,
        distortion_k1=camera_in.distortion_k1,
        distortion_k2=camera_in.distortion_k2,
        distortion_k3=camera_in.distortion_k3,
        distortion_k4=camera_in.distortion_k4,
        distortion_k5=camera_in.distortion_k5,
        distortion_k6=camera_in.distortion_k6,
        distortion_p1=camera_in.distortion_p1,
        distortion_p2=camera_in.distortion_p2,
        distortion_poly_order=camera_in.distortion_poly_order,
        distortion_poly_has_forward=bool(camera_in.distortion_poly_has_forward),
        distortion_poly_has_inverse=bool(camera_in.distortion_poly_has_inv),
        distortion_poly_forward_u=tuple(forward_u),
        distortion_poly_forward_v=tuple(forward_v),
        distortion_poly_inverse_u=tuple(inverse_u),
        distortion_poly_inverse_v=tuple(inverse_v),
        coord_sys=camera_in.coord_sys,
        subpixel_center_map=camera_in.subpixel_center_map,
        psf_type=camera_in.psf_type,
        psf_sigma_x=camera_in.psf_sigma_x,
        psf_sigma_y=camera_in.psf_sigma_y,
        psf_theta=camera_in.psf_theta,
        psf_support_rad=camera_in.psf_supp_rad,
        psf_separable=camera_in.psf_separable,
    )


@cython.cfunc
def _make_raster_config(config: Any) -> cr.CRasterConfig:
    config_out: cr.CRasterConfig
    config_out.render_mode = int(config.render_mode)
    config_out.total_threads = int(config.total_threads)
    config_out.frame_batch_size_per_group = int(
        config.frame_batch_size_per_group,
    )
    config_out.max_geom_jobs_in_flight_per_group = int(
        config.max_geom_jobs_in_flight_per_group,
    )
    config_out.max_geom_workers_per_job = int(config.max_geom_workers_per_job)
    config_out.geom_scheduling_mode = int(config.geom_scheduling_mode)
    config_out.max_raster_workers_per_job = int(
        config.max_raster_workers_per_job,
    )
    config_out.save_strategy = int(config.save_strategy)
    config_out.image_save_mode = int(config.image_save_mode)
    config_out.hull_mode = int(config.hull_mode)
    config_out.newton_seed_mode = int(config.newton_seed_mode)
    config_out.newton_seed_reuse = int(config.newton_seed_reuse)
    config_out.report = int(config.report)
    config_out.tile_size_min = int(config.tile_size_min)
    config_out.tile_size_max = int(config.tile_size_max)
    config_out.background_value = float(config.background_value)
    config_out.disk_save_overlap = 1 if config.disk_save_overlap else 0
    config_out.tile_size_override = int(config.tile_size_override)
    config_out.save_frame_buff_count = int(config.save_frame_buffer_count)
    config_out.save_format = int(config.save_format)
    config_out.save_bits = int(config.save_bits)
    config_out.save_scaling = int(config.save_scaling)
    config_out.save_scaling_min = float(config.save_scaling_min)
    config_out.save_scaling_max = float(config.save_scaling_max)
    config_out.full_stats_save_solver_csv = int(
        config.full_stats_save_solver_csv,
    )
    config_out.full_stats_save_iter_map = int(
        config.full_stats_save_iter_map,
    )
    config_out.full_stats_save_xi_map = int(
        config.full_stats_save_xi_map,
    )
    config_out.full_stats_save_eta_map = int(
        config.full_stats_save_eta_map,
    )
    config_out.full_stats_save_conv_map = int(
        config.full_stats_save_conv_map,
    )
    config_out.full_stats_save_jac_det_map = int(
        config.full_stats_save_jac_det_map,
    )
    config_out.full_stats_save_tile_timing_map = int(
        config.full_stats_save_tile_timing_map,
    )
    config_out.full_stats_save_tile_density_map = int(
        config.full_stats_save_tile_density_map,
    )
    config_out.full_stats_save_tile_occupancy_map = int(
        config.full_stats_save_tile_occupancy_map,
    )
    config_out.full_stats_save_depth_map = int(
        config.full_stats_save_depth_map,
    )
    config_out.full_stats_save_earlyout_map = int(
        config.full_stats_save_earlyout_map,
    )
    config_out.full_stats_save_pixel_occupancy_map = int(
        config.full_stats_save_pixel_occupancy_map,
    )
    config_out.full_stats_save_normals_map = int(
        config.full_stats_save_normals_map,
    )
    return config_out


@cython.cfunc
def _make_func_params(params_in: Any) -> cr.CFuncShaderParams:
    params_out: cr.CFuncShaderParams
    params_out.coord_scale_0 = float(params_in.coord_scale[0])
    params_out.coord_scale_1 = float(params_in.coord_scale[1])
    params_out.coord_offset_0 = float(params_in.coord_offset[0])
    params_out.coord_offset_1 = float(params_in.coord_offset[1])
    params_out.output_scale = float(params_in.output_scale)
    params_out.output_offset = float(params_in.output_offset)
    params_out.constant_value = float(params_in.constant_value)
    params_out.constant_value_rgb_0 = float(params_in.constant_value_rgb[0])
    params_out.constant_value_rgb_1 = float(params_in.constant_value_rgb[1])
    params_out.constant_value_rgb_2 = float(params_in.constant_value_rgb[2])
    params_out.linear_coeff_0 = float(params_in.linear_coeffs[0])
    params_out.linear_coeff_1 = float(params_in.linear_coeffs[1])
    params_out.linear_coeff_2 = float(params_in.linear_coeffs[2])
    params_out.linear_coeff_rgb_00 = float(params_in.linear_coeffs_rgb[0][0])
    params_out.linear_coeff_rgb_01 = float(params_in.linear_coeffs_rgb[0][1])
    params_out.linear_coeff_rgb_02 = float(params_in.linear_coeffs_rgb[0][2])
    params_out.linear_coeff_rgb_10 = float(params_in.linear_coeffs_rgb[1][0])
    params_out.linear_coeff_rgb_11 = float(params_in.linear_coeffs_rgb[1][1])
    params_out.linear_coeff_rgb_12 = float(params_in.linear_coeffs_rgb[1][2])
    params_out.linear_coeff_rgb_20 = float(params_in.linear_coeffs_rgb[2][0])
    params_out.linear_coeff_rgb_21 = float(params_in.linear_coeffs_rgb[2][1])
    params_out.linear_coeff_rgb_22 = float(params_in.linear_coeffs_rgb[2][2])
    params_out.quadratic_coeff_0 = float(params_in.quadratic_coeffs[0])
    params_out.quadratic_coeff_1 = float(params_in.quadratic_coeffs[1])
    params_out.quadratic_coeff_2 = float(params_in.quadratic_coeffs[2])
    params_out.quadratic_coeff_3 = float(params_in.quadratic_coeffs[3])
    params_out.quadratic_coeff_4 = float(params_in.quadratic_coeffs[4])
    params_out.quadratic_coeff_5 = float(params_in.quadratic_coeffs[5])
    params_out.quadratic_coeff_rgb_00 = float(
        params_in.quadratic_coeffs_rgb[0][0],
    )
    params_out.quadratic_coeff_rgb_01 = float(
        params_in.quadratic_coeffs_rgb[0][1],
    )
    params_out.quadratic_coeff_rgb_02 = float(
        params_in.quadratic_coeffs_rgb[0][2],
    )
    params_out.quadratic_coeff_rgb_03 = float(
        params_in.quadratic_coeffs_rgb[0][3],
    )
    params_out.quadratic_coeff_rgb_04 = float(
        params_in.quadratic_coeffs_rgb[0][4],
    )
    params_out.quadratic_coeff_rgb_05 = float(
        params_in.quadratic_coeffs_rgb[0][5],
    )
    params_out.quadratic_coeff_rgb_10 = float(
        params_in.quadratic_coeffs_rgb[1][0],
    )
    params_out.quadratic_coeff_rgb_11 = float(
        params_in.quadratic_coeffs_rgb[1][1],
    )
    params_out.quadratic_coeff_rgb_12 = float(
        params_in.quadratic_coeffs_rgb[1][2],
    )
    params_out.quadratic_coeff_rgb_13 = float(
        params_in.quadratic_coeffs_rgb[1][3],
    )
    params_out.quadratic_coeff_rgb_14 = float(
        params_in.quadratic_coeffs_rgb[1][4],
    )
    params_out.quadratic_coeff_rgb_15 = float(
        params_in.quadratic_coeffs_rgb[1][5],
    )
    params_out.quadratic_coeff_rgb_20 = float(
        params_in.quadratic_coeffs_rgb[2][0],
    )
    params_out.quadratic_coeff_rgb_21 = float(
        params_in.quadratic_coeffs_rgb[2][1],
    )
    params_out.quadratic_coeff_rgb_22 = float(
        params_in.quadratic_coeffs_rgb[2][2],
    )
    params_out.quadratic_coeff_rgb_23 = float(
        params_in.quadratic_coeffs_rgb[2][3],
    )
    params_out.quadratic_coeff_rgb_24 = float(
        params_in.quadratic_coeffs_rgb[2][4],
    )
    params_out.quadratic_coeff_rgb_25 = float(
        params_in.quadratic_coeffs_rgb[2][5],
    )
    params_out.wave_num_scalar_0 = float(params_in.wave_num_scalar[0])
    params_out.wave_num_scalar_1 = float(params_in.wave_num_scalar[1])
    params_out.wave_num_rgb_0 = float(params_in.wave_num_rgb[0])
    params_out.wave_num_rgb_1 = float(params_in.wave_num_rgb[1])
    params_out.wave_num_rgb_2 = float(params_in.wave_num_rgb[2])
    params_out.sinusoidal_bias = float(params_in.sinusoidal_bias)
    params_out.sinusoidal_amp_0 = float(params_in.sinusoidal_amplitudes[0])
    params_out.sinusoidal_amp_1 = float(params_in.sinusoidal_amplitudes[1])
    params_out.sinusoidal_bias_rgb_0 = float(
        params_in.sinusoidal_bias_rgb[0],
    )
    params_out.sinusoidal_bias_rgb_1 = float(
        params_in.sinusoidal_bias_rgb[1],
    )
    params_out.sinusoidal_bias_rgb_2 = float(
        params_in.sinusoidal_bias_rgb[2],
    )
    params_out.sinusoidal_amp_rgb_0 = float(
        params_in.sinusoidal_amplitudes_rgb[0],
    )
    params_out.sinusoidal_amp_rgb_1 = float(
        params_in.sinusoidal_amplitudes_rgb[1],
    )
    params_out.sinusoidal_amp_rgb_2 = float(
        params_in.sinusoidal_amplitudes_rgb[2],
    )
    params_out.checker_level_0 = float(params_in.checker_levels[0])
    params_out.checker_level_1 = float(params_in.checker_levels[1])
    params_out.checker_smooth_frequency = float(
        params_in.checker_smooth_frequency,
    )
    params_out.lambertian_coeff_0 = float(params_in.lambertian_coeffs[0])
    params_out.lambertian_coeff_1 = float(params_in.lambertian_coeffs[1])
    params_out.lambertian_coeff_rgb_00 = float(
        params_in.lambertian_coeffs_rgb[0][0],
    )
    params_out.lambertian_coeff_rgb_01 = float(
        params_in.lambertian_coeffs_rgb[0][1],
    )
    params_out.lambertian_coeff_rgb_10 = float(
        params_in.lambertian_coeffs_rgb[1][0],
    )
    params_out.lambertian_coeff_rgb_11 = float(
        params_in.lambertian_coeffs_rgb[1][1],
    )
    params_out.lambertian_coeff_rgb_20 = float(
        params_in.lambertian_coeffs_rgb[2][0],
    )
    params_out.lambertian_coeff_rgb_21 = float(
        params_in.lambertian_coeffs_rgb[2][1],
    )
    params_out.eggbox_mean = float(params_in.eggbox_mean)
    params_out.eggbox_contrast = float(params_in.eggbox_contrast)
    params_out.eggbox_pitch_0 = float(params_in.eggbox_pitch[0])
    params_out.eggbox_pitch_1 = float(params_in.eggbox_pitch[1])
    params_out.eggbox_phase_0 = float(params_in.eggbox_phase[0])
    params_out.eggbox_phase_1 = float(params_in.eggbox_phase[1])
    params_out.extra_0 = float(params_in.extra[0])
    params_out.extra_1 = float(params_in.extra[1])
    params_out.extra_2 = float(params_in.extra[2])
    params_out.extra_3 = float(params_in.extra[3])
    return params_out


@cython.cfunc
def _make_array_2d_f64(
    view_in: cython.double[:, ::1],
    rows_num: cython.Py_ssize_t,
    cols_num: cython.Py_ssize_t,
) -> cr.CArray2DF64:
    return cr.CArray2DF64(
        cython.address(view_in[0, 0]),
        rows_num,
        cols_num,
    )


@cython.cfunc
def _make_array_2d_usize(
    view_in: cython.size_t[:, ::1],
    rows_num: cython.Py_ssize_t,
    cols_num: cython.Py_ssize_t,
) -> cr.CArray2DUsize:
    return cr.CArray2DUsize(
        cython.address(view_in[0, 0]),
        rows_num,
        cols_num,
    )


@cython.cfunc
def _make_array_3d_f64(
    view_in: cython.double[:, :, ::1],
    dim0: cython.Py_ssize_t,
    dim1: cython.Py_ssize_t,
    dim2: cython.Py_ssize_t,
) -> cr.CArray3DF64:
    return cr.CArray3DF64(
        cython.address(view_in[0, 0, 0]),
        dim0,
        dim1,
        dim2,
    )


@cython.cfunc
def _make_array_3d_u8(
    view_in: cython.uchar[:, :, ::1],
    dim0: cython.Py_ssize_t,
    dim1: cython.Py_ssize_t,
    dim2: cython.Py_ssize_t,
) -> cr.CArray3DU8:
    return cr.CArray3DU8(cython.address(view_in[0, 0, 0]), dim0, dim1, dim2)


@cython.cfunc
def _make_array_3d_u16(
    view_in: cython.ushort[:, :, ::1],
    dim0: cython.Py_ssize_t,
    dim1: cython.Py_ssize_t,
    dim2: cython.Py_ssize_t,
) -> cr.CArray3DU16:
    return cr.CArray3DU16(cython.address(view_in[0, 0, 0]), dim0, dim1, dim2)


@cython.cfunc
def _empty_array_2d_f64() -> cr.CArray2DF64:
    return cr.CArray2DF64(cython.NULL, 0, 0)


@cython.cfunc
def _empty_array_2d_usize() -> cr.CArray2DUsize:
    return cr.CArray2DUsize(cython.NULL, 0, 0)


@cython.cfunc
def _empty_array_3d_f64() -> cr.CArray3DF64:
    return cr.CArray3DF64(cython.NULL, 0, 0, 0)


@cython.cfunc
def _empty_array_3d_u8() -> cr.CArray3DU8:
    return cr.CArray3DU8(cython.NULL, 0, 0, 0)


@cython.cfunc
def _empty_array_3d_u16() -> cr.CArray3DU16:
    return cr.CArray3DU16(cython.NULL, 0, 0, 0)


def _last_error_message() -> str:
    uint8_buf_np = np.zeros((512,), dtype=np.uint8)
    uint8_view: cython.uchar[::1] = uint8_buf_np
    cr.rileyGetLastError(cython.address(uint8_view[0]), uint8_view.shape[0])
    return bytes(uint8_buf_np).split(b"\0", 1)[0].decode("utf-8")


def _raise_last_error() -> None:
    msg = _last_error_message()
    if msg:
        raise RuntimeError(msg)
    raise RuntimeError("riley wrapper call failed")


def _as_shape_2d(array_in: Any) -> tuple[int, int]:
    if getattr(array_in, "ndim", None) != 2:
        raise ValueError("expected a 2D numpy array")
    return int(array_in.shape[0]), int(array_in.shape[1])


def _as_shape_3d(array_in: Any) -> tuple[int, int, int]:
    if getattr(array_in, "ndim", None) != 3:
        raise ValueError("expected a 3D numpy array")
    return (
        int(array_in.shape[0]),
        int(array_in.shape[1]),
        int(array_in.shape[2]),
    )


def _normalize_meshes(meshes_in: Any) -> list[Any]:
    if isinstance(meshes_in, (list, tuple)):
        return list(meshes_in)
    return [meshes_in]


def _normalize_cameras(cameras_in: Any) -> list[Any]:
    if isinstance(cameras_in, (list, tuple)):
        return list(cameras_in)
    return [cameras_in]


def _contig_f64_2d(array_in: Any, label: str) -> np.ndarray:
    array_np = np.ascontiguousarray(array_in, dtype=np.float64)
    _as_shape_2d(array_np)
    return array_np


def _contig_u_size_2d(array_in: Any) -> np.ndarray:
    array_np = np.ascontiguousarray(array_in, dtype=np.uintp)
    _as_shape_2d(array_np)
    return array_np


def _contig_f64_3d(array_in: Any, label: str) -> np.ndarray:
    array_np = np.ascontiguousarray(array_in, dtype=np.float64)
    _as_shape_3d(array_np)
    return array_np


def _contig_texture(
    texture_in: Any,
    channels_num: int,
    storage: int,
) -> np.ndarray:
    texture_base = np.asarray(texture_in)
    texture_np = np.array(texture_base, copy=True, order="C")
    if getattr(texture_np, "ndim", None) == 2:
        if channels_num != 1:
            raise ValueError("rgb texture must have shape (3, rows, cols)")
        texture_np = np.ascontiguousarray(texture_np[None, :, :])
    if getattr(texture_np, "ndim", None) != 3:
        raise ValueError("texture must have shape (channels, rows, cols)")
    if int(texture_np.shape[0]) != channels_num:
        raise ValueError("texture channel count does not match shader type")
    if storage == int(TextureStorage.u8):
        if texture_np.dtype != np.uint8:
            raise ValueError("u8 texture storage requires a uint8 array")
    elif storage == int(TextureStorage.u16):
        if texture_np.dtype != np.uint16:
            raise ValueError("u16 texture storage requires a uint16 array")
    elif storage == int(TextureStorage.floating):
        if texture_np.dtype not in (np.float32, np.float64):
            raise ValueError("floating texture storage requires a float32 or float64 array")
        texture_np = np.ascontiguousarray(texture_np, dtype=np.float64)
    else:
        raise ValueError("unsupported texture storage")
    return texture_np


@cython.boundscheck(False)
@cython.wraparound(False)
def roi_cent_from_coords(coords_in: Any) -> tuple[float, float, float]:
    coords_np = _contig_f64_2d(coords_in, "coords")
    rows_num, cols_num = _as_shape_2d(coords_np)
    if cols_num != 3:
        raise ValueError("coords must have shape (N, 3)")

    coords_view: cython.double[:, ::1] = coords_np
    coords_c = _make_array_2d_f64(coords_view, rows_num, cols_num)
    out_cent: cr.CVec3F64

    if cr.rileyRoiCentFromCoords(
        cython.address(coords_c),
        cython.address(out_cent),
    ) != 0:
        _raise_last_error()

    return (out_cent.x, out_cent.y, out_cent.z)


@cython.boundscheck(False)
@cython.wraparound(False)
def pos_fill_frame_from_rot(
    coords_in: Any,
    pixels_num: tuple[int, int],
    pixels_size: tuple[float, float],
    focal_length: float,
    rot_world: tuple[float, float, float],
    frame_fill: float = 1.0,
) -> tuple[float, float, float]:
    coords_np = _contig_f64_2d(coords_in, "coords")
    rows_num, cols_num = _as_shape_2d(coords_np)
    if cols_num != 3:
        raise ValueError("coords must have shape (N, 3)")

    coords_view: cython.double[:, ::1] = coords_np
    coords_c = _make_array_2d_f64(coords_view, rows_num, cols_num)
    out_pos: cr.CVec3F64

    if cr.rileyPosFillFrameFromRot(
        cython.address(coords_c),
        _make_cvec2_u32(tuple(pixels_num)),
        _make_cvec2_f64(tuple(pixels_size)),
        float(focal_length),
        _make_cvec3(tuple(rot_world)),
        float(frame_fill),
        cython.address(out_pos),
    ) != 0:
        _raise_last_error()

    return (out_pos.x, out_pos.y, out_pos.z)


@cython.cfunc
def _fill_mesh_array(
    mesh_list: list[Any],
    mesh_array: cython.pointer[cr.CMeshInput],
    keepalive: list[Any],
) -> None:
    nn: cython.size_t
    for nn in range(len(mesh_list)):
        mesh = mesh_list[nn]
        coords_np = _contig_f64_2d(mesh.coords, "coords")
        connect_np = _contig_u_size_2d(mesh.connect)
        coords_shape = _as_shape_2d(coords_np)
        if coords_shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")

        coords_view: cython.double[:, ::1] = coords_np
        connect_view: cython.size_t[:, ::1] = connect_np
        mesh_array[nn].mesh_type = int(mesh.mesh_type)
        mesh_array[nn].coords = _make_array_2d_f64(
            coords_view,
            coords_np.shape[0],
            coords_np.shape[1],
        )
        mesh_array[nn].connect = _make_array_2d_usize(
            connect_view,
            connect_np.shape[0],
            connect_np.shape[1],
        )

        if mesh.disp is None:
            mesh_array[nn].disp = _empty_array_3d_f64()
        else:
            disp_np = _contig_f64_3d(mesh.disp, "disp")
            disp_shape = _as_shape_3d(disp_np)
            disp_view: cython.double[:, :, ::1] = disp_np
            mesh_array[nn].disp = _make_array_3d_f64(
                disp_view,
                disp_shape[0],
                disp_shape[1],
                disp_shape[2],
            )
            keepalive.append(disp_np)

        shader_tag = int(mesh.shader_type)
        mesh_array[nn].shader_tag = shader_tag
        texture_storage = int(mesh.texture_storage)
        mesh_array[nn].texture_storage = texture_storage
        mesh_array[nn].sample = int(mesh.sample)
        mesh_array[nn].sample_mode = int(mesh.sample_mode)
        mesh_array[nn].bits = int(mesh.bits)
        mesh_array[nn].scaling_tag = int(mesh.scaling_type)
        mesh_array[nn].scaling_min = float(mesh.scaling_min)
        mesh_array[nn].scaling_max = float(mesh.scaling_max)
        mesh_array[nn].scale_over = int(mesh.scale_over)
        mesh_array[nn].func_shader_builtin = int(mesh.func_shader_builtin)
        mesh_array[nn].func_shader_coord_mode = int(mesh.func_shader_coord_mode)
        mesh_array[nn].func_shader_params = _make_func_params(
            mesh.func_shader_params,
        )
        mesh_array[nn].normal_type = int(mesh.normal_type)

        if mesh.uvs is None:
            mesh_array[nn].uvs = _empty_array_2d_f64()
        else:
            uvs_np = _contig_f64_2d(mesh.uvs, "uvs")
            uvs_shape = _as_shape_2d(uvs_np)
            uvs_view: cython.double[:, ::1] = uvs_np
            mesh_array[nn].uvs = _make_array_2d_f64(
                uvs_view,
                uvs_shape[0],
                uvs_shape[1],
            )
            keepalive.append(uvs_np)

        texture_channels = 0
        if shader_tag == int(ShaderType.tex):
            texture_channels = 1
        elif shader_tag == int(ShaderType.tex_rgb):
            texture_channels = 3

        if mesh.texture is None:
            mesh_array[nn].tex = _empty_array_3d_f64()
            mesh_array[nn].tex_u8 = _empty_array_3d_u8()
            mesh_array[nn].tex_u16 = _empty_array_3d_u16()
        else:
            if texture_channels == 0:
                raise ValueError("texture provided for non-texture shader")
            texture_np = _contig_texture(
                mesh.texture,
                texture_channels,
                texture_storage,
            )
            texture_shape = _as_shape_3d(texture_np)
            mesh_array[nn].tex = _empty_array_3d_f64()
            mesh_array[nn].tex_u8 = _empty_array_3d_u8()
            mesh_array[nn].tex_u16 = _empty_array_3d_u16()
            if texture_storage == int(TextureStorage.u8):
                texture_view_u8: cython.uchar[:, :, ::1] = texture_np
                mesh_array[nn].tex_u8 = _make_array_3d_u8(
                    texture_view_u8,
                    texture_shape[0], texture_shape[1], texture_shape[2],
                )
            elif texture_storage == int(TextureStorage.u16):
                texture_view_u16: cython.ushort[:, :, ::1] = texture_np
                mesh_array[nn].tex_u16 = _make_array_3d_u16(
                    texture_view_u16,
                    texture_shape[0], texture_shape[1], texture_shape[2],
                )
            else:
                texture_view_f: cython.double[:, :, ::1] = texture_np
                mesh_array[nn].tex = _make_array_3d_f64(
                    texture_view_f,
                    texture_shape[0], texture_shape[1], texture_shape[2],
                )
            keepalive.append(texture_np)

        if mesh.nodal_field is None:
            mesh_array[nn].nodal_field = _empty_array_3d_f64()
        else:
            nodal_field_np = _contig_f64_3d(mesh.nodal_field, "nodal_field")
            nodal_shape = _as_shape_3d(nodal_field_np)
            if (
                shader_tag == int(ShaderType.nodal_rgb)
                and nodal_shape[2] != 3
            ):
                raise ValueError(
                    "nodal_rgb field must have shape (time, nodes, 3)",
                )
            nodal_view: cython.double[:, :, ::1] = nodal_field_np
            mesh_array[nn].nodal_field = _make_array_3d_f64(
                nodal_view,
                nodal_shape[0],
                nodal_shape[1],
                nodal_shape[2],
            )
            keepalive.append(nodal_field_np)

        keepalive.append(coords_np)
        keepalive.append(connect_np)


@cython.boundscheck(False)
@cython.wraparound(False)
def roi_cent_over_meshes(meshes: Any) -> tuple[float, float, float]:
    mesh_list = _normalize_meshes(meshes)
    meshes_len: cython.size_t = len(mesh_list)
    mesh_array = cython.cast(
        cython.pointer[cr.CMeshInput],
        malloc(meshes_len * cython.sizeof(cr.CMeshInput)),
    )
    out_cent: cr.CVec3F64
    keepalive: list[Any] = []
    image_c: cr.CImageBuffF64
    if mesh_array == cython.NULL:
        raise MemoryError()
    try:
        _fill_mesh_array(mesh_list, mesh_array, keepalive)
        if cr.rileyRoiCentOverMeshes(
            mesh_array,
            meshes_len,
            cython.address(out_cent),
        ) != 0:
            _raise_last_error()
    finally:
        free(mesh_array)
    return (out_cent.x, out_cent.y, out_cent.z)


@cython.boundscheck(False)
@cython.wraparound(False)
def pos_fill_frame_from_rot_over_meshes(
    meshes: Any,
    pixels_num: tuple[int, int],
    pixels_size: tuple[float, float],
    focal_length: float,
    rot_world: tuple[float, float, float],
    frame_fill: float = 1.0,
) -> tuple[float, float, float]:
    mesh_list = _normalize_meshes(meshes)
    meshes_len: cython.size_t = len(mesh_list)
    mesh_array = cython.cast(
        cython.pointer[cr.CMeshInput],
        malloc(meshes_len * cython.sizeof(cr.CMeshInput)),
    )
    out_pos: cr.CVec3F64
    keepalive: list[Any] = []
    if mesh_array == cython.NULL:
        raise MemoryError()
    try:
        _fill_mesh_array(mesh_list, mesh_array, keepalive)
        if cr.rileyPosFillFrameFromRotOverMeshes(
            mesh_array,
            meshes_len,
            _make_cvec2_u32(tuple(pixels_num)),
            _make_cvec2_f64(tuple(pixels_size)),
            float(focal_length),
            _make_cvec3(tuple(rot_world)),
            float(frame_fill),
            cython.address(out_pos),
        ) != 0:
            _raise_last_error()
    finally:
        free(mesh_array)
    return (out_pos.x, out_pos.y, out_pos.z)


def save_stereo_pair(
    out_dir: str,
    stereo_file_name: str,
    camera_0: Camera,
    camera_1: Camera,
) -> None:
    cam0_c = _make_camera_input(camera_0)
    cam1_c = _make_camera_input(camera_1)
    out_dir_bytes = out_dir.encode("utf-8")
    file_name_bytes = stereo_file_name.encode("utf-8")
    if cr.rileySaveStereoPair(
        out_dir_bytes,
        file_name_bytes,
        cython.address(cam0_c),
        cython.address(cam1_c),
    ) != 0:
        _raise_last_error()


def save_camera(
    out_dir: str,
    file_name: str,
    camera_idx: int,
    camera: Camera,
) -> None:
    camera_c = _make_camera_input(camera)
    out_dir_bytes = out_dir.encode("utf-8")
    file_name_bytes = file_name.encode("utf-8")
    if cr.rileySaveCamera(
        out_dir_bytes,
        file_name_bytes,
        camera_idx,
        cython.address(camera_c),
    ) != 0:
        _raise_last_error()


def load_camera(
    dir_path: str,
    file_name: str,
) -> Camera:
    dir_bytes = dir_path.encode("utf-8")
    file_name_bytes = file_name.encode("utf-8")
    camera_c: cr.CCameraInput
    if cr.rileyLoadCamera(
        dir_bytes,
        file_name_bytes,
        cython.address(camera_c),
    ) != 0:
        _raise_last_error()
    return _camera_input_from_c(camera_c)


def load_stereo_pair(
    dir_path: str,
    stereo_file_name: str,
) -> tuple[Camera, Camera]:
    dir_bytes = dir_path.encode("utf-8")
    file_bytes = stereo_file_name.encode("utf-8")
    cam0_c: cr.CCameraInput
    cam1_c: cr.CCameraInput
    if cr.rileyLoadStereoPair(
        dir_bytes,
        file_bytes,
        cython.address(cam0_c),
        cython.address(cam1_c),
    ) != 0:
        _raise_last_error()
    return (
        _camera_input_from_c(cam0_c),
        _camera_input_from_c(cam1_c),
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def raster(
    meshes: Any,
    cameras: Any,
    config: RasterConfig,
    out_dir: str | None = None,
) -> np.ndarray | None:
    mesh_list = _normalize_meshes(meshes)
    camera_list = _normalize_cameras(cameras)
    meshes_len: cython.size_t = len(mesh_list)
    cameras_len: cython.size_t = len(camera_list)
    mesh_array = cython.cast(
        cython.pointer[cr.CMeshInput],
        malloc(meshes_len * cython.sizeof(cr.CMeshInput)),
    )
    camera_array = cython.cast(
        cython.pointer[cr.CCameraInput],
        malloc(cameras_len * cython.sizeof(cr.CCameraInput)),
    )
    config_c: cr.CRasterConfig = _make_raster_config(config)
    image_np: np.ndarray | None = None
    image_ptr: cython.pointer[cr.CImageBuffF64] = cython.cast(
        cython.pointer[cr.CImageBuffF64],
        cython.NULL,
    )
    image_c: cr.CImageBuffF64
    keepalive: list[Any] = []

    if mesh_array == cython.NULL or camera_array == cython.NULL:
        if mesh_array != cython.NULL:
            free(mesh_array)
        if camera_array != cython.NULL:
            free(camera_array)
        raise MemoryError()

    try:
        _fill_mesh_array(mesh_list, mesh_array, keepalive)

        nn: cython.size_t
        for nn in range(cameras_len):
            camera_array[nn] = _make_camera_input(camera_list[nn])

        out_dir_ptr: cython.p_char = cython.cast(cython.p_char, cython.NULL)
        if out_dir is not None:
            out_dir_bytes: bytes = out_dir.encode("utf-8")
            out_dir_ptr = out_dir_bytes
            keepalive.append(out_dir_bytes)

        if config.save_strategy in (SaveStrategy.memory, SaveStrategy.both):
            dims_c: cr.CDims5Usize
            if cr.rileyCalcOutputDimsScene(
                mesh_array,
                meshes_len,
                camera_array,
                cameras_len,
                cython.address(config_c),
                cython.address(dims_c),
            ) != 0:
                _raise_last_error()

            image_np = np.empty(
                (
                    dims_c.dim0,
                    dims_c.dim1,
                    dims_c.dim2,
                    dims_c.dim3,
                    dims_c.dim4,
                ),
                dtype=np.float64,
            )
            image_view: cython.double[:, :, :, :, ::1] = image_np
            image_c.elems = cython.address(image_view[0, 0, 0, 0, 0])
            image_c.dims = dims_c
            image_ptr = cython.address(image_c)

        if cr.rileyRaster(
            mesh_array,
            meshes_len,
            camera_array,
            cameras_len,
            cython.address(config_c),
            out_dir_ptr,
            image_ptr,
        ) != 0:
            _raise_last_error()
    finally:
        free(mesh_array)
        free(camera_array)

    return image_np





__all__ = [
    "Camera",
    "CameraInput",
    "CameraCoordSys",
    "HullMode",
    "Mesh",
    "MeshInput",
    "MeshType",
    "NewtonSeedMode",
    "NewtonSeedReuse",
    "NormalType",
    "GeometrySchedulingMode",
    "ImageSaveMode",
    "RasterConfig",
    "RenderMode",
    "ReportMode",
    "SaveStrategy",
    "ScaleOver",
    "ScaleStrategy",
    "ShaderType",
    "TextureStorage",
    "SubPixelCenterMap",
    "FuncShaderBuiltin",
    "FuncCoordMode",
    "FuncShaderParams",
    "TextureSample",
    "TextureSampleMode",
    "PsfType",
    "ImageFormat",
    "load_camera",
    "load_stereo_pair",
    "pos_fill_frame_from_rot",
    "pos_fill_frame_from_rot_over_meshes",
    "raster",
    "roi_cent_from_coords",
    "roi_cent_over_meshes",
    "save_camera",
    "save_stereo_pair",
]
