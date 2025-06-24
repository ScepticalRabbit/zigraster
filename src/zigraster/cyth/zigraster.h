#ifndef ZIGRASTER_H
#define ZIGRASTER_H

#include <stddef.h>
#include <stdint.h>

typedef struct cVec2U32 {
    uint32_t x;
    uint32_t y;
} CVec2U32;

typedef struct cVec2F {
    double x;
    double y;
} CVec2F;

typedef struct cVec3F {
    double x;
    double y;
    double z;
} CVec3F;

typedef struct cMat44F {
    double* mat;
    size_t numel;
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

void printCamera(const CCamera* cam);

#endif // ZIGRASTER_H