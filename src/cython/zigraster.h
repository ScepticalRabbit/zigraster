#ifndef ZIGRASTER_H
#define ZIGRASTER_H

#include <stddef.h>
#include <stdint.h>

typedef struct CVec2U32 {
    uint32_t x;
    uint32_t y;
} CVec2U32;

typedef struct CVec2F {
    double x;
    double y;
} CVec2F;

typedef struct CVec3F {
    double x;
    double y;
    double z;
} CVec2F;

typedef struct CMat44F {
    double* mat;
    const size_t numel;
} CMat44F;

typedef struct CCamera {
    CVec2U32 pixels_num;
    CVec2F pixels_size;
    CVec3F pos_world;
    CVec3F rot_world;
    CVec3F roi_cent_world;
    uint8_t subsample;
    CVec2F sensor_size;
    CVec2F image_dims;
    double image_dist;
    CMat44F cam_to_world;
    CMat44F world_to_cam;
} CCamera;


#endif // ZIGRASTER_H