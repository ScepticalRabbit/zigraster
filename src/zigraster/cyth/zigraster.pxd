from libc.stdint cimport uint32_t, uint8_t

cdef extern from "zigraster.h":

    ctypedef struct CVec2U32:
        uint32_t x
        uint32_t y

    ctypedef struct CVec2F:
        double x
        double y

    ctypedef struct CVec3F:
        double x
        double y
        double z

    ctypedef struct CMat44F:
        double* mat
        size_t numel

    ctypedef struct CCamera:
        CVec2U32 pixels_num
        CVec2F pixels_size
        CVec3F pos_world
        CVec3F rot_world
        CVec3F roi_cent_world
        uint8_t subsample
        CVec2F sensor_size
        CVec2F image_dims
        double image_dist
        CMat44F cam_to_world
        CMat44F world_to_cam

    void printCamera(const CCamera* cam)

