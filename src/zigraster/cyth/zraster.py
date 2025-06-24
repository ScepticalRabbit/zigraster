import cython
from cython.cimports import zigraster as zr
import numpy as np
import pyvale as pyv

def set_camera(cam: pyv.CameraData) -> None:
    pixels_num: zr.CVec2U32 = zr.CVec2U32(cam.pixels_num[0],cam.pixels_num[1])
    pixels_size: zr.CVec2F = zr.CVec2F(cam.pixels_size[0],cam.pixels_size[1])
    pos_world: zr.CVec3F = zr.CVec3F(cam.pos_world[0],
                                     cam.pos_world[1],
                                     cam.pos_world[2])
    rot_angs = cam.rot_world.as_euler("zyx",degrees=False)
    rot_world: zr.CVec3F = zr.CVec3F(rot_angs[0],rot_angs[1],rot_angs[2])
    roi_cent_world: zr.CVec3F = zr.CVec3F(cam.roi_cent_world[0],
                                          cam.roi_cent_world[1],
                                          cam.roi_cent_world[2])
    sub_sample: cython.ushort = cam.sub_samp
    sensor_size: zr.CVec2F = zr.CVec2F(cam.sensor_size[0],
                                       cam.sensor_size[1])
    image_dims: zr.CVec2F = zr.CVec2F(cam.image_dims[0],
                                      cam.image_dims[1])
    image_dist: cython.double = cam.image_dist

    c_to_w_flat_np = np.ascontiguousarray(cam.cam_to_world_mat.flatten())
    c_to_w: cython.double[::1] = c_to_w_flat_np
    cam_to_world_mat: zr.CMat44F = zr.CMat44F(cython.address(c_to_w[0]),16)

    w_to_c_flat_np = np.ascontiguousarray(cam.world_to_cam_mat.flatten())
    w_to_c: cython.double[::1] = w_to_c_flat_np
    world_to_cam_mat: zr.CMat44F = zr.CMat44F(cython.address(w_to_c[0]),16)

    ccam: zr.CCamera = zr.CCamera(
        pixels_num,
        pixels_size,
        pos_world,
        rot_world,
        roi_cent_world,
        sub_sample,
        sensor_size,
        image_dims,
        image_dist,
        cam_to_world_mat,
        world_to_cam_mat,
    )

    # print()
    # print("Cython: Camera")
    # print(f"{ccam}")
    # print()

    zr.printCamera(cython.address(ccam))
