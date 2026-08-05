#ifndef RILEY_H
#define RILEY_H

#include <stddef.h>
#include <stdint.h>

/*
 * Public Riley C ABI Contract
 *
 * The public Riley C ABI is fixed to the production Riley build:
 * - precision: f64
 * - SIMD: on
 *
 * This keeps the exported ABI stable for C, Cython and Python callers.
 */

typedef struct c_vec2_u32 {
    uint32_t x;
    uint32_t y;
} CVec2U32;

typedef struct c_vec2_f64 {
    double x;
    double y;
} CVec2F64;

typedef struct c_vec3_f64 {
    double x;
    double y;
    double z;
} CVec3F64;

typedef struct c_array_2d_f64 {
    const double* elems;
    size_t rows_num;
    size_t cols_num;
} CArray2DF64;

typedef struct c_array_2d_usize {
    const size_t* elems;
    size_t rows_num;
    size_t cols_num;
} CArray2DUsize;

typedef struct c_array_3d_f64 {
    const double* elems;
    size_t dim0;
    size_t dim1;
    size_t dim2;
} CArray3DF64;

typedef struct c_array_3d_u8 {
    const uint8_t* elems;
    size_t dim0;
    size_t dim1;
    size_t dim2;
} CArray3DU8;

typedef struct c_array_3d_u16 {
    const uint16_t* elems;
    size_t dim0;
    size_t dim1;
    size_t dim2;
} CArray3DU16;

typedef struct c_dims_5_usize {
    size_t dim0;
    size_t dim1;
    size_t dim2;
    size_t dim3;
    size_t dim4;
} CDims5Usize;

typedef struct c_image_buff_f64 {
    double* elems;
    CDims5Usize dims;
} CImageBuffF64;

typedef struct c_camera_input {
    CVec2U32 pixels_num;
    CVec2F64 pixels_size;
    CVec3F64 pos_world;
    CVec3F64 rot_world;
    CVec3F64 roi_cent_world;
    double focal_length;
    uint32_t sub_sample;
    uint32_t distortion_model;
    double distortion_k1;
    double distortion_k2;
    double distortion_k3;
    double distortion_k4;
    double distortion_k5;
    double distortion_k6;
    double distortion_p1;
    double distortion_p2;
    uint32_t distortion_poly_order;
    uint8_t distortion_poly_has_forward;
    uint8_t distortion_poly_has_inv;
    double distortion_poly_forward_u[10];
    double distortion_poly_forward_v[10];
    double distortion_poly_inv_u[10];
    double distortion_poly_inv_v[10];
    uint32_t coord_sys;
    uint32_t subpixel_center_map;
    uint32_t psf_type;
    double psf_sigma_x;
    double psf_sigma_y;
    double psf_theta;
    double psf_supp_rad;
    uint32_t psf_separable;
} CCameraInput;

typedef struct c_func_shader_params {
    double coord_scale_0;
    double coord_scale_1;
    double coord_offset_0;
    double coord_offset_1;
    double output_scale;
    double output_offset;
    double constant_value;
    double constant_value_rgb_0;
    double constant_value_rgb_1;
    double constant_value_rgb_2;
    double linear_coeff_0;
    double linear_coeff_1;
    double linear_coeff_2;
    double linear_coeff_rgb_00;
    double linear_coeff_rgb_01;
    double linear_coeff_rgb_02;
    double linear_coeff_rgb_10;
    double linear_coeff_rgb_11;
    double linear_coeff_rgb_12;
    double linear_coeff_rgb_20;
    double linear_coeff_rgb_21;
    double linear_coeff_rgb_22;
    double quadratic_coeff_0;
    double quadratic_coeff_1;
    double quadratic_coeff_2;
    double quadratic_coeff_3;
    double quadratic_coeff_4;
    double quadratic_coeff_5;
    double quadratic_coeff_rgb_00;
    double quadratic_coeff_rgb_01;
    double quadratic_coeff_rgb_02;
    double quadratic_coeff_rgb_03;
    double quadratic_coeff_rgb_04;
    double quadratic_coeff_rgb_05;
    double quadratic_coeff_rgb_10;
    double quadratic_coeff_rgb_11;
    double quadratic_coeff_rgb_12;
    double quadratic_coeff_rgb_13;
    double quadratic_coeff_rgb_14;
    double quadratic_coeff_rgb_15;
    double quadratic_coeff_rgb_20;
    double quadratic_coeff_rgb_21;
    double quadratic_coeff_rgb_22;
    double quadratic_coeff_rgb_23;
    double quadratic_coeff_rgb_24;
    double quadratic_coeff_rgb_25;
    double wave_num_scalar_0;
    double wave_num_scalar_1;
    double wave_num_rgb_0;
    double wave_num_rgb_1;
    double wave_num_rgb_2;
    double sinusoidal_bias;
    double sinusoidal_amp_0;
    double sinusoidal_amp_1;
    double sinusoidal_bias_rgb_0;
    double sinusoidal_bias_rgb_1;
    double sinusoidal_bias_rgb_2;
    double sinusoidal_amp_rgb_0;
    double sinusoidal_amp_rgb_1;
    double sinusoidal_amp_rgb_2;
    double checker_level_0;
    double checker_level_1;
    double checker_smooth_frequency;
    double lambertian_coeff_0;
    double lambertian_coeff_1;
    double lambertian_coeff_rgb_00;
    double lambertian_coeff_rgb_01;
    double lambertian_coeff_rgb_10;
    double lambertian_coeff_rgb_11;
    double lambertian_coeff_rgb_20;
    double lambertian_coeff_rgb_21;
    double eggbox_mean;
    double eggbox_contrast;
    double eggbox_pitch_0;
    double eggbox_pitch_1;
    double eggbox_phase_0;
    double eggbox_phase_1;
    double extra_0;
    double extra_1;
    double extra_2;
    double extra_3;
} CFuncShaderParams;


typedef struct c_mesh_input {
    uint32_t mesh_type;
    CArray2DF64 coords;
    CArray2DUsize connect;
    CArray3DF64 disp;
    uint32_t shader_tag;
    CArray2DF64 uvs;
    CArray3DF64 tex;
    CArray3DU8 tex_u8;
    CArray3DU16 tex_u16;
    uint32_t texture_storage;
    uint32_t sample;
    uint32_t sample_mode;
    int bits;
    uint32_t scaling_tag;
    double scaling_min;
    double scaling_max;
    CArray3DF64 nodal_field;
    uint32_t scale_over;
    uint32_t func_shader_builtin;
    uint32_t func_shader_coord_mode;
    CFuncShaderParams func_shader_params;
    uint32_t normal_type;
} CMeshInput;

typedef struct c_raster_config {
    uint32_t render_mode;
    uint16_t total_threads;
    uint16_t frame_batch_size_per_group;
    uint16_t max_geom_jobs_in_flight_per_group;
    uint16_t max_geom_workers_per_job;
    uint32_t geom_scheduling_mode;
    uint16_t max_raster_workers_per_job;
    uint32_t save_strategy;
    uint32_t image_save_mode;
    uint32_t hull_mode;
    uint32_t newton_seed_mode;
    uint32_t newton_seed_reuse;
    uint32_t report;
    uint16_t tile_size_min;
    uint16_t tile_size_max;
    double background_value;
    uint8_t disk_save_overlap;
    uint16_t tile_size_override;
    size_t save_frame_buff_count;
    uint32_t save_format;
    uint32_t save_bits;
    uint32_t save_scaling;
    double save_scaling_min;
    double save_scaling_max;
    uint8_t full_stats_save_solver_csv;
    uint8_t full_stats_save_iter_map;
    uint8_t full_stats_save_xi_map;
    uint8_t full_stats_save_eta_map;
    uint8_t full_stats_save_conv_map;
    uint8_t full_stats_save_jac_det_map;
    uint8_t full_stats_save_tile_timing_map;
    uint8_t full_stats_save_tile_density_map;
    uint8_t full_stats_save_tile_occupancy_map;
    uint8_t full_stats_save_depth_map;
    uint8_t full_stats_save_earlyout_map;
    uint8_t full_stats_save_pixel_occupancy_map;
    uint8_t full_stats_save_normals_map;
} CRasterConfig;

size_t rileyGetLastError(uint8_t* out_buf, size_t out_buf_len);

int rileyRoiCentFromCoords(
    const CArray2DF64* in_coords,
    CVec3F64* out_cent
);

int rileyPosFillFrameFromRot(
    const CArray2DF64* in_coords,
    CVec2U32 pixels_num,
    CVec2F64 pixels_size,
    double focal_length,
    CVec3F64 rot_world,
    double frame_fill,
    CVec3F64* out_pos
);

int rileyRoiCentOverMeshes(
    const CMeshInput* in_meshes,
    size_t meshes_len,
    CVec3F64* out_cent
);

int rileyPosFillFrameFromRotOverMeshes(
    const CMeshInput* in_meshes,
    size_t meshes_len,
    CVec2U32 pixels_num,
    CVec2F64 pixels_size,
    double focal_length,
    CVec3F64 rot_world,
    double frame_fill,
    CVec3F64* out_pos
);

int rileyCalcOutputDimsScene(
    const CMeshInput* in_meshes,
    size_t meshes_len,
    const CCameraInput* in_cameras,
    size_t cameras_len,
    const CRasterConfig* in_config,
    CDims5Usize* out_dims
);

int rileySaveCamera(
    const char* out_dir_path,
    const char* file_name,
    size_t camera_idx,
    const CCameraInput* camera_in
);

int rileyLoadCamera(
    const char* dir_path,
    const char* file_name,
    CCameraInput* camera_out
);

int rileySaveStereoPair(
    const char* out_dir_path,
    const char* stereo_file_name,
    const CCameraInput* cam0_in,
    const CCameraInput* cam1_in
);

int rileyLoadStereoPair(
    const char* dir_path,
    const char* stereo_file_name,
    CCameraInput* cam0_out,
    CCameraInput* cam1_out
);

int rileyRaster(
    const CMeshInput* in_meshes,
    size_t meshes_len,
    const CCameraInput* in_cameras,
    size_t cameras_len,
    const CRasterConfig* in_config,
    const char* out_dir_path,
    CImageBuffF64* out_image
);

#endif
