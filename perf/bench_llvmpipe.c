#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stddef.h>
#include <unistd.h>

#define RUNS 25
#define VIEW_WIDTH 1600
#define VIEW_HEIGHT 1000
#ifndef SSAA_SAMPLES
#define SSAA_SAMPLES 4
#endif
#ifndef OUT_TAG
#define OUT_TAG "ssaa2"
#endif
#define PI 3.14159265358979323846

// Struct to store vertex information
typedef struct {
    float pos[3];
    float uv[2];
    float param[2];
    float field_g;
    float field_rgb[3];
} Vertex;

// Stats container matching Riley's structure
typedef struct {
    double median;
    double mad;
    double min;
    double max;
    double cov;
} MetricStats;

typedef struct {
    char case_name[64];
    char element[16];
    char shader[32];
    char interpolator[32];
    MetricStats clear_time_ms;
    MetricStats draw_time_ms;
    MetricStats resolve_time_ms;
    MetricStats time_ms;
} CaseResult;

static const float RILEY_LINEAR_COEFFS[3] = {
    0.5f, 0.25f, 0.2f
};

static const float RILEY_LINEAR_COEFFS_RGB[9] = {
    0.5f, 0.25f, 0.0f,
    0.5f, 0.0f, 0.25f,
    0.5f, 0.15f, -0.15f
};

static const float RILEY_QUADRATIC_COEFFS[6] = {
    0.35f, 0.2f, 0.15f, 0.1f, -0.08f, 0.06f
};

static const float RILEY_QUADRATIC_COEFFS_RGB[18] = {
    0.3f, 0.0f, 0.0f, 0.2f, 0.0f, 0.0f,
    0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f,
    0.3f, 0.0f, 0.0f, 0.0f, 0.12f, 0.0f
};

void rename_if_exists(const char* src_path, const char* dst_path) {
    if (access(src_path, F_OK) != 0) {
        return;
    }
    if (access(dst_path, F_OK) == 0) {
        return;
    }
    if (rename(src_path, dst_path) != 0) {
        fprintf(stderr, "Warning: failed to rename %s to %s\n",
                src_path, dst_path);
    }
}

void force_llvmpipe_single_thread(void) {
    setenv("LIBGL_ALWAYS_SOFTWARE", "1", 1);
    setenv("GALLIUM_DRIVER", "llvmpipe", 1);
    setenv("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe", 1);
    setenv("LP_NUM_THREADS", "0", 1);
    setenv("EGL_PLATFORM", "surfaceless", 1);
}

void make_timestamp_string(char* buffer, size_t buffer_size) {
    time_t now = time(NULL);
    struct tm now_tm;
    localtime_r(&now, &now_tm);
    strftime(buffer, buffer_size, "%Y%m%d_%H%M%S", &now_tm);
}

// Helper to load BMP files into a flat float array (pixels in range [0, 255])
float* load_bmp(const char* filename, int* width_out, int* height_out, int channels) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open BMP: %s\n", filename);
        return NULL;
    }

    unsigned char header[14];
    if (fread(header, 1, 14, f) != 14) {
        fclose(f);
        return NULL;
    }

    if (header[0] != 'B' || header[1] != 'M') {
        fprintf(stderr, "Not a BMP file: %s\n", filename);
        fclose(f);
        return NULL;
    }

    unsigned int offset = *(unsigned int*)&header[10];
    unsigned int dib_size;
    if (fread(&dib_size, 4, 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    int width = 0, height = 0;
    unsigned short bit_count = 0;
    unsigned int compression = 0;

    if (dib_size >= 40) {
        fread(&width, 4, 1, f);
        fread(&height, 4, 1, f);
        fseek(f, 2, SEEK_CUR); // skip planes
        fread(&bit_count, 2, 1, f);
        fread(&compression, 4, 1, f);
        if (compression != 0) {
            fprintf(stderr, "Compressed BMP not supported: %s\n", filename);
            fclose(f);
            return NULL;
        }
    } else {
        fprintf(stderr, "Unsupported DIB header: %u in %s\n", dib_size, filename);
        fclose(f);
        return NULL;
    }

    int abs_height = height < 0 ? -height : height;
    int abs_width = width < 0 ? -width : width;
    *width_out = abs_width;
    *height_out = abs_height;

    float* img_data = malloc(abs_width * abs_height * channels * sizeof(float));
    if (!img_data) {
        fclose(f);
        return NULL;
    }

    if (bit_count == 24) {
        fseek(f, offset, SEEK_SET);
        int row_padding = (4 - (abs_width * 3) % 4) % 4;
        for (int y = 0; y < abs_height; ++y) {
            int r = (height > 0) ? (abs_height - 1 - y) : y;
            for (int x = 0; x < abs_width; ++x) {
                unsigned char bgr[3];
                fread(bgr, 1, 3, f);
                if (channels == 3) {
                    img_data[(r * abs_width + x) * 3 + 0] = (float)bgr[2]; // R
                    img_data[(r * abs_width + x) * 3 + 1] = (float)bgr[1]; // G
                    img_data[(r * abs_width + x) * 3 + 2] = (float)bgr[0]; // B
                } else if (channels == 1) {
                    float val = 0.299f * bgr[2] + 0.587f * bgr[1] + 0.114f * bgr[0];
                    img_data[r * abs_width + x] = val;
                }
            }
            fseek(f, row_padding, SEEK_CUR);
        }
    } else if (bit_count == 8) {
        fseek(f, 14 + dib_size, SEEK_SET);
        int palette_size = (offset - (14 + dib_size)) / 4;
        unsigned char (*palette)[4] = malloc(palette_size * 4);
        fread(palette, 4, palette_size, f);

        fseek(f, offset, SEEK_SET);
        int row_padding = (4 - abs_width % 4) % 4;
        for (int y = 0; y < abs_height; ++y) {
            int r = (height > 0) ? (abs_height - 1 - y) : y;
            for (int x = 0; x < abs_width; ++x) {
                unsigned char index;
                fread(&index, 1, 1, f);
                unsigned char* color = palette[index];
                if (channels == 3) {
                    img_data[(r * abs_width + x) * 3 + 0] = (float)color[2]; // R
                    img_data[(r * abs_width + x) * 3 + 1] = (float)color[1]; // G
                    img_data[(r * abs_width + x) * 3 + 2] = (float)color[0]; // B
                } else if (channels == 1) {
                    float val = 0.299f * color[2] + 0.587f * color[1] + 0.114f * color[0];
                    img_data[r * abs_width + x] = val;
                }
            }
            fseek(f, row_padding, SEEK_CUR);
        }
        free(palette);
    } else {
        fprintf(stderr, "Unsupported bit count %d in %s\n", bit_count, filename);
        free(img_data);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return img_data;
}

// Shader Compilation Helpers
GLuint compile_shader(GLenum type, const char* source) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &source, NULL);
    glCompileShader(s);
    GLint success;
    glGetShaderiv(s, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(s, 512, NULL, info);
        fprintf(stderr, "Shader compile error: %s\n", info);
        return 0;
    }
    return s;
}

GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint success;
    glGetProgramiv(p, GL_LINK_STATUS, &success);
    if (!success) {
        char info[512];
        glGetProgramInfoLog(p, 512, NULL, info);
        fprintf(stderr, "Program link error: %s\n", info);
        return 0;
    }
    return p;
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

MetricStats calc_metric_stats(double* data, int n) {
    double* sorted = malloc(n * sizeof(double));
    memcpy(sorted, data, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_doubles);

    double median;
    int mid = n / 2;
    if (n % 2 == 0) {
        median = (sorted[mid - 1] + sorted[mid]) / 2.0;
    } else {
        median = sorted[mid];
    }

    double* abs_devs = malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        abs_devs[i] = fabs(sorted[i] - median);
    }
    qsort(abs_devs, n, sizeof(double), compare_doubles);

    double mad;
    if (n % 2 == 0) {
        mad = (abs_devs[mid - 1] + abs_devs[mid]) / 2.0;
    } else {
        mad = abs_devs[mid];
    }

    double min = sorted[0];
    double max = sorted[n - 1];
    double cov = median == 0.0 ? 0.0 : (mad / median * 100.0);

    free(sorted);
    free(abs_devs);

    return (MetricStats){median, mad, min, max, cov};
}

// GLSL Shader Source Strings with Dynamic Projection and Uniform variables
const char* vs_nodal_grey = 
    "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 3) in float aNodalField;\n"
    "uniform mat4 uViewMatrix;\n"
    "uniform mat4 uProjMatrix;\n"
    "out float vNodalField;\n"
    "void main() {\n"
    "    gl_Position = uProjMatrix * uViewMatrix * vec4(aPos, 1.0);\n"
    "    vNodalField = aNodalField;\n"
    "}\n";

const char* fs_nodal_grey = 
    "#version 330 core\n"
    "in float vNodalField;\n"
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    fragColor = vNodalField;\n"
    "}\n";

const char* vs_nodal_rgb = 
    "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 4) in vec3 aNodalFieldRGB;\n"
    "uniform mat4 uViewMatrix;\n"
    "uniform mat4 uProjMatrix;\n"
    "out vec3 vNodalFieldRGB;\n"
    "void main() {\n"
    "    gl_Position = uProjMatrix * uViewMatrix * vec4(aPos, 1.0);\n"
    "    vNodalFieldRGB = aNodalFieldRGB;\n"
    "}\n";

const char* fs_nodal_rgb = 
    "#version 330 core\n"
    "in vec3 vNodalFieldRGB;\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "void main() {\n"
    "    fragColor = vNodalFieldRGB;\n"
    "}\n";

const char* vs_uv = 
    "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 1) in vec2 aUV;\n"
    "uniform mat4 uViewMatrix;\n"
    "uniform mat4 uProjMatrix;\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "    gl_Position = uProjMatrix * uViewMatrix * vec4(aPos, 1.0);\n"
    "    vUV = aUV;\n"
    "}\n";

const char* vs_param = 
    "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 2) in vec2 aParam;\n"
    "uniform mat4 uViewMatrix;\n"
    "uniform mat4 uProjMatrix;\n"
    "out vec2 vParam;\n"
    "void main() {\n"
    "    gl_Position = uProjMatrix * uViewMatrix * vec4(aPos, 1.0);\n"
    "    vParam = aParam;\n"
    "}\n";

const char* fs_tex8_grey_linear = 
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D uTexture;\n"
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    fragColor = texture(uTexture, vUV).r;\n"
    "}\n";

const char* fs_tex8_rgb_linear = 
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D uTexture;\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "void main() {\n"
    "    fragColor = texture(uTexture, vUV).rgb;\n"
    "}\n";

const char* fs_tex8_grey_cubic = 
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D uTexture;\n"
    "uniform vec2 uTextureSize;\n"
    "layout(location = 0) out float fragColor;\n"
    "float cubicCoeffCatmullRom(float x) {\n"
    "    float abs_x = abs(x);\n"
    "    if (abs_x <= 1.0) {\n"
    "        return ((1.5 * abs_x - 2.5) * abs_x + 0.0) * abs_x + 1.0;\n"
    "    } else if (abs_x < 2.0) {\n"
    "        return ((-0.5 * abs_x + 2.5) * abs_x - 4.0) * abs_x + 2.0;\n"
    "    }\n"
    "    return 0.0;\n"
    "}\n"
    "void main() {\n"
    "    vec2 tex_coords = vUV * (uTextureSize - 1.0);\n"
    "    vec2 tex_i = floor(tex_coords);\n"
    "    vec2 tex_frac = tex_coords - tex_i;\n"
    "    float coeffs_x[4];\n"
    "    coeffs_x[0] = cubicCoeffCatmullRom(tex_frac.x + 1.0);\n"
    "    coeffs_x[1] = cubicCoeffCatmullRom(tex_frac.x);\n"
    "    coeffs_x[2] = cubicCoeffCatmullRom(tex_frac.x - 1.0);\n"
    "    coeffs_x[3] = cubicCoeffCatmullRom(tex_frac.x - 2.0);\n"
    "    float coeffs_y[4];\n"
    "    coeffs_y[0] = cubicCoeffCatmullRom(tex_frac.y + 1.0);\n"
    "    coeffs_y[1] = cubicCoeffCatmullRom(tex_frac.y);\n"
    "    coeffs_y[2] = cubicCoeffCatmullRom(tex_frac.y - 1.0);\n"
    "    coeffs_y[3] = cubicCoeffCatmullRom(tex_frac.y - 2.0);\n"
    "    float sum_val = 0.0;\n"
    "    float sum_weight = 0.0;\n"
    "    for (int jj = 0; jj < 4; ++jj) {\n"
    "        for (int ii = 0; ii < 4; ++ii) {\n"
    "            float w = coeffs_x[ii] * coeffs_y[jj];\n"
    "            vec2 sample_pixel = tex_i + vec2(ii - 1, jj - 1);\n"
    "            sample_pixel = clamp(sample_pixel, vec2(0.0), uTextureSize - 1.0);\n"
    "            float texel = texelFetch(uTexture, ivec2(sample_pixel), 0).r;\n"
    "            sum_val += texel * w;\n"
    "            sum_weight += w;\n"
    "        }\n"
    "    }\n"
    "    float inv_sum = abs(sum_weight) < 1e-5 ? 1.0 : 1.0 / sum_weight;\n"
    "    fragColor = sum_val * inv_sum;\n"
    "}\n";

const char* fs_tex8_rgb_cubic = 
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D uTexture;\n"
    "uniform vec2 uTextureSize;\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "float cubicCoeffCatmullRom(float x) {\n"
    "    float abs_x = abs(x);\n"
    "    if (abs_x <= 1.0) {\n"
    "        return ((1.5 * abs_x - 2.5) * abs_x + 0.0) * abs_x + 1.0;\n"
    "    } else if (abs_x < 2.0) {\n"
    "        return ((-0.5 * abs_x + 2.5) * abs_x - 4.0) * abs_x + 2.0;\n"
    "    }\n"
    "    return 0.0;\n"
    "}\n"
    "void main() {\n"
    "    vec2 tex_coords = vUV * (uTextureSize - 1.0);\n"
    "    vec2 tex_i = floor(tex_coords);\n"
    "    vec2 tex_frac = tex_coords - tex_i;\n"
    "    float coeffs_x[4];\n"
    "    coeffs_x[0] = cubicCoeffCatmullRom(tex_frac.x + 1.0);\n"
    "    coeffs_x[1] = cubicCoeffCatmullRom(tex_frac.x);\n"
    "    coeffs_x[2] = cubicCoeffCatmullRom(tex_frac.x - 1.0);\n"
    "    coeffs_x[3] = cubicCoeffCatmullRom(tex_frac.x - 2.0);\n"
    "    float coeffs_y[4];\n"
    "    coeffs_y[0] = cubicCoeffCatmullRom(tex_frac.y + 1.0);\n"
    "    coeffs_y[1] = cubicCoeffCatmullRom(tex_frac.y);\n"
    "    coeffs_y[2] = cubicCoeffCatmullRom(tex_frac.y - 1.0);\n"
    "    coeffs_y[3] = cubicCoeffCatmullRom(tex_frac.y - 2.0);\n"
    "    vec3 sum_val = vec3(0.0);\n"
    "    float sum_weight = 0.0;\n"
    "    for (int jj = 0; jj < 4; ++jj) {\n"
    "        for (int ii = 0; ii < 4; ++ii) {\n"
    "            float w = coeffs_x[ii] * coeffs_y[jj];\n"
    "            vec2 sample_pixel = tex_i + vec2(ii - 1, jj - 1);\n"
    "            sample_pixel = clamp(sample_pixel, vec2(0.0), uTextureSize - 1.0);\n"
    "            vec3 texel = texelFetch(uTexture, ivec2(sample_pixel), 0).rgb;\n"
    "            sum_val += texel * w;\n"
    "            sum_weight += w;\n"
    "        }\n"
    "    }\n"
    "    float inv_sum = abs(sum_weight) < 1e-5 ? 1.0 : 1.0 / sum_weight;\n"
    "    fragColor = sum_val * inv_sum;\n"
    "}\n";

const char* fs_tex8_grey_quintic = 
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D uTexture;\n"
    "uniform vec2 uTextureSize;\n"
    "layout(location = 0) out float fragColor;\n"
    "float quinticBSplineCoeff(float x) {\n"
    "    float r = abs(x);\n"
    "    if (r >= 3.0) return 0.0;\n"
    "    if (r <= 1.0) {\n"
    "        return ((((-(1.0 / 12.0) * r + (1.0 / 4.0)) * r + 0.0) * r - (1.0 / 2.0)) * r + 0.0) * r + (11.0 / 20.0);\n"
    "    } else if (r <= 2.0) {\n"
    "        float t = r - 1.0;\n"
    "        return (((((1.0 / 24.0) * t - (1.0 / 6.0)) * t + (1.0 / 6.0)) * t + (1.0 / 6.0)) * t - (5.0 / 12.0)) * t + (13.0 / 60.0);\n"
    "    } else {\n"
    "        float u = r - 2.0;\n"
    "        return (((((-(1.0 / 120.0) * u + (1.0 / 24.0)) * u - (1.0 / 12.0)) * u + (1.0 / 12.0)) * u - (1.0 / 24.0)) * u + (1.0 / 120.0));\n"
    "    }\n"
    "}\n"
    "void main() {\n"
    "    vec2 tex_coords = vUV * (uTextureSize - 1.0);\n"
    "    vec2 tex_i = floor(tex_coords);\n"
    "    vec2 tex_frac = tex_coords - tex_i;\n"
    "    float coeffs_x[6];\n"
    "    coeffs_x[0] = quinticBSplineCoeff(tex_frac.x + 2.0);\n"
    "    coeffs_x[1] = quinticBSplineCoeff(tex_frac.x + 1.0);\n"
    "    coeffs_x[2] = quinticBSplineCoeff(tex_frac.x);\n"
    "    coeffs_x[3] = quinticBSplineCoeff(tex_frac.x - 1.0);\n"
    "    coeffs_x[4] = quinticBSplineCoeff(tex_frac.x - 2.0);\n"
    "    coeffs_x[5] = quinticBSplineCoeff(tex_frac.x - 3.0);\n"
    "    float coeffs_y[6];\n"
    "    coeffs_y[0] = quinticBSplineCoeff(tex_frac.y + 2.0);\n"
    "    coeffs_y[1] = quinticBSplineCoeff(tex_frac.y + 1.0);\n"
    "    coeffs_y[2] = quinticBSplineCoeff(tex_frac.y);\n"
    "    coeffs_y[3] = quinticBSplineCoeff(tex_frac.y - 1.0);\n"
    "    coeffs_y[4] = quinticBSplineCoeff(tex_frac.y - 2.0);\n"
    "    coeffs_y[5] = quinticBSplineCoeff(tex_frac.y - 3.0);\n"
    "    float sum_val = 0.0;\n"
    "    float sum_weight = 0.0;\n"
    "    for (int jj = 0; jj < 6; ++jj) {\n"
    "        for (int ii = 0; ii < 6; ++ii) {\n"
    "            float w = coeffs_x[ii] * coeffs_y[jj];\n"
    "            vec2 sample_pixel = tex_i + vec2(ii - 2, jj - 2);\n"
    "            sample_pixel = clamp(sample_pixel, vec2(0.0), uTextureSize - 1.0);\n"
    "            float texel = texelFetch(uTexture, ivec2(sample_pixel), 0).r;\n"
    "            sum_val += texel * w;\n"
    "            sum_weight += w;\n"
    "        }\n"
    "    }\n"
    "    float inv_sum = abs(sum_weight) < 1e-5 ? 1.0 : 1.0 / sum_weight;\n"
    "    fragColor = sum_val * inv_sum;\n"
    "}\n";

const char* fs_tex8_rgb_quintic = 
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform sampler2D uTexture;\n"
    "uniform vec2 uTextureSize;\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "float quinticBSplineCoeff(float x) {\n"
    "    float r = abs(x);\n"
    "    if (r >= 3.0) return 0.0;\n"
    "    if (r <= 1.0) {\n"
    "        return ((((-(1.0 / 12.0) * r + (1.0 / 4.0)) * r + 0.0) * r - (1.0 / 2.0)) * r + 0.0) * r + (11.0 / 20.0);\n"
    "    } else if (r <= 2.0) {\n"
    "        float t = r - 1.0;\n"
    "        return (((((1.0 / 24.0) * t - (1.0 / 6.0)) * t + (1.0 / 6.0)) * t + (1.0 / 6.0)) * t - (5.0 / 12.0)) * t + (13.0 / 60.0);\n"
    "    } else {\n"
    "        float u = r - 2.0;\n"
    "        return (((((-(1.0 / 120.0) * u + (1.0 / 24.0)) * u - (1.0 / 12.0)) * u + (1.0 / 12.0)) * u - (1.0 / 24.0)) * u + (1.0 / 120.0));\n"
    "    }\n"
    "}\n"
    "void main() {\n"
    "    vec2 tex_coords = vUV * (uTextureSize - 1.0);\n"
    "    vec2 tex_i = floor(tex_coords);\n"
    "    vec2 tex_frac = tex_coords - tex_i;\n"
    "    float coeffs_x[6];\n"
    "    coeffs_x[0] = quinticBSplineCoeff(tex_frac.x + 2.0);\n"
    "    coeffs_x[1] = quinticBSplineCoeff(tex_frac.x + 1.0);\n"
    "    coeffs_x[2] = quinticBSplineCoeff(tex_frac.x);\n"
    "    coeffs_x[3] = quinticBSplineCoeff(tex_frac.x - 1.0);\n"
    "    coeffs_x[4] = quinticBSplineCoeff(tex_frac.x - 2.0);\n"
    "    coeffs_x[5] = quinticBSplineCoeff(tex_frac.x - 3.0);\n"
    "    float coeffs_y[6];\n"
    "    coeffs_y[0] = quinticBSplineCoeff(tex_frac.y + 2.0);\n"
    "    coeffs_y[1] = quinticBSplineCoeff(tex_frac.y + 1.0);\n"
    "    coeffs_y[2] = quinticBSplineCoeff(tex_frac.y);\n"
    "    coeffs_y[3] = quinticBSplineCoeff(tex_frac.y - 1.0);\n"
    "    coeffs_y[4] = quinticBSplineCoeff(tex_frac.y - 2.0);\n"
    "    coeffs_y[5] = quinticBSplineCoeff(tex_frac.y - 3.0);\n"
    "    vec3 sum_val = vec3(0.0);\n"
    "    float sum_weight = 0.0;\n"
    "    for (int jj = 0; jj < 6; ++jj) {\n"
    "        for (int ii = 0; ii < 6; ++ii) {\n"
    "            float w = coeffs_x[ii] * coeffs_y[jj];\n"
    "            vec2 sample_pixel = tex_i + vec2(ii - 2, jj - 2);\n"
    "            sample_pixel = clamp(sample_pixel, vec2(0.0), uTextureSize - 1.0);\n"
    "            vec3 texel = texelFetch(uTexture, ivec2(sample_pixel), 0).rgb;\n"
    "            sum_val += texel * w;\n"
    "            sum_weight += w;\n"
    "        }\n"
    "    }\n"
    "    float inv_sum = abs(sum_weight) < 1e-5 ? 1.0 : 1.0 / sum_weight;\n"
    "    fragColor = sum_val * inv_sum;\n"
    "}\n";

const char* fs_const_grey = 
    "#version 330 core\n"
    "uniform float uConstantColor;\n"
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    fragColor = uConstantColor;\n"
    "}\n";

const char* fs_const_rgb = 
    "#version 330 core\n"
    "uniform vec3 uConstantColorRGB;\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "void main() {\n"
    "    fragColor = uConstantColorRGB;\n"
    "}\n";

const char* fs_linear_param_grey =
    "#version 330 core\n"
    "in vec2 vParam;\n"
    "uniform float uLinearCoeffs[3];\n"
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    fragColor = uLinearCoeffs[0] +\n"
    "        uLinearCoeffs[1] * vParam.x +\n"
    "        uLinearCoeffs[2] * vParam.y;\n"
    "}\n";

const char* fs_linear_uv_grey =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform float uLinearCoeffs[3];\n"
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    fragColor = uLinearCoeffs[0] +\n"
    "        uLinearCoeffs[1] * vUV.x +\n"
    "        uLinearCoeffs[2] * vUV.y;\n"
    "}\n";

const char* fs_linear_param_rgb =
    "#version 330 core\n"
    "in vec2 vParam;\n"
    "uniform float uLinearCoeffsRGB[9];\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "void main() {\n"
    "    float r = uLinearCoeffsRGB[0] +\n"
    "        uLinearCoeffsRGB[1] * vParam.x +\n"
    "        uLinearCoeffsRGB[2] * vParam.y;\n"
    "    float g = uLinearCoeffsRGB[3] +\n"
    "        uLinearCoeffsRGB[4] * vParam.x +\n"
    "        uLinearCoeffsRGB[5] * vParam.y;\n"
    "    float b = uLinearCoeffsRGB[6] +\n"
    "        uLinearCoeffsRGB[7] * vParam.x +\n"
    "        uLinearCoeffsRGB[8] * vParam.y;\n"
    "    fragColor = vec3(r, g, b);\n"
    "}\n";

const char* fs_linear_uv_rgb =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform float uLinearCoeffsRGB[9];\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "void main() {\n"
    "    float r = uLinearCoeffsRGB[0] +\n"
    "        uLinearCoeffsRGB[1] * vUV.x +\n"
    "        uLinearCoeffsRGB[2] * vUV.y;\n"
    "    float g = uLinearCoeffsRGB[3] +\n"
    "        uLinearCoeffsRGB[4] * vUV.x +\n"
    "        uLinearCoeffsRGB[5] * vUV.y;\n"
    "    float b = uLinearCoeffsRGB[6] +\n"
    "        uLinearCoeffsRGB[7] * vUV.x +\n"
    "        uLinearCoeffsRGB[8] * vUV.y;\n"
    "    fragColor = vec3(r, g, b);\n"
    "}\n";

const char* fs_quadratic_param_grey =
    "#version 330 core\n"
    "in vec2 vParam;\n"
    "uniform float uQuadraticCoeffs[6];\n"
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    float coord_u = vParam.x;\n"
    "    float coord_v = vParam.y;\n"
    "    float term_u = coord_u *\n"
    "        (uQuadraticCoeffs[1] + uQuadraticCoeffs[3] * coord_u);\n"
    "    float term_v = coord_v *\n"
    "        (uQuadraticCoeffs[2] +\n"
    "            uQuadraticCoeffs[4] * coord_u +\n"
    "            uQuadraticCoeffs[5] * coord_v);\n"
    "    fragColor = uQuadraticCoeffs[0] + term_u + term_v;\n"
    "}\n";

const char* fs_quadratic_uv_grey =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform float uQuadraticCoeffs[6];\n"
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    float coord_u = vUV.x;\n"
    "    float coord_v = vUV.y;\n"
    "    float term_u = coord_u *\n"
    "        (uQuadraticCoeffs[1] + uQuadraticCoeffs[3] * coord_u);\n"
    "    float term_v = coord_v *\n"
    "        (uQuadraticCoeffs[2] +\n"
    "            uQuadraticCoeffs[4] * coord_u +\n"
    "            uQuadraticCoeffs[5] * coord_v);\n"
    "    fragColor = uQuadraticCoeffs[0] + term_u + term_v;\n"
    "}\n";

const char* fs_quadratic_param_rgb =
    "#version 330 core\n"
    "in vec2 vParam;\n"
    "uniform float uQuadraticCoeffsRGB[18];\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "float evalQuadratic(int base_idx, float coord_u, float coord_v) {\n"
    "    float term_u = coord_u *\n"
    "        (uQuadraticCoeffsRGB[base_idx + 1] +\n"
    "            uQuadraticCoeffsRGB[base_idx + 3] * coord_u);\n"
    "    float term_v = coord_v *\n"
    "        (uQuadraticCoeffsRGB[base_idx + 2] +\n"
    "            uQuadraticCoeffsRGB[base_idx + 4] * coord_u +\n"
    "            uQuadraticCoeffsRGB[base_idx + 5] * coord_v);\n"
    "    return uQuadraticCoeffsRGB[base_idx + 0] + term_u + term_v;\n"
    "}\n"
    "void main() {\n"
    "    fragColor = vec3(\n"
    "        evalQuadratic(0, vParam.x, vParam.y),\n"
    "        evalQuadratic(6, vParam.x, vParam.y),\n"
    "        evalQuadratic(12, vParam.x, vParam.y)\n"
    "    );\n"
    "}\n";

const char* fs_quadratic_uv_rgb =
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform float uQuadraticCoeffsRGB[18];\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "float evalQuadratic(int base_idx, float coord_u, float coord_v) {\n"
    "    float term_u = coord_u *\n"
    "        (uQuadraticCoeffsRGB[base_idx + 1] +\n"
    "            uQuadraticCoeffsRGB[base_idx + 3] * coord_u);\n"
    "    float term_v = coord_v *\n"
    "        (uQuadraticCoeffsRGB[base_idx + 2] +\n"
    "            uQuadraticCoeffsRGB[base_idx + 4] * coord_u +\n"
    "            uQuadraticCoeffsRGB[base_idx + 5] * coord_v);\n"
    "    return uQuadraticCoeffsRGB[base_idx + 0] + term_u + term_v;\n"
    "}\n"
    "void main() {\n"
    "    fragColor = vec3(\n"
    "        evalQuadratic(0, vUV.x, vUV.y),\n"
    "        evalQuadratic(6, vUV.x, vUV.y),\n"
    "        evalQuadratic(12, vUV.x, vUV.y)\n"
    "    );\n"
    "}\n";

const char* fs_sin_param_grey = 
    "#version 330 core\n"
    "in vec2 vParam;\n"
    "uniform float uWaveNum;\n"
    "uniform vec3 uWaveCoeffs;\n" // x=bias, y=sin_scale, z=cos_scale
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    fragColor = uWaveCoeffs.x + uWaveCoeffs.y * sin(uWaveNum * vParam.x) + uWaveCoeffs.z * cos(uWaveNum * vParam.y);\n"
    "}\n";

const char* fs_sin_uv_grey = 
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform float uWaveNum;\n"
    "uniform vec3 uWaveCoeffs;\n"
    "layout(location = 0) out float fragColor;\n"
    "void main() {\n"
    "    fragColor = uWaveCoeffs.x + uWaveCoeffs.y * sin(uWaveNum * vUV.x) + uWaveCoeffs.z * cos(uWaveNum * vUV.y);\n"
    "}\n";

const char* fs_sin_param_rgb = 
    "#version 330 core\n"
    "in vec2 vParam;\n"
    "uniform float uWaveNum;\n"
    "uniform vec3 uWaveCoeffsR;\n"
    "uniform vec3 uWaveCoeffsB;\n"
    "uniform vec3 uWaveCoeffsG;\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "void main() {\n"
    "    float r = uWaveCoeffsR.x + uWaveCoeffsR.y * sin(uWaveNum * vParam.x);\n"
    "    float g = uWaveCoeffsG.x + uWaveCoeffsG.y * cos(uWaveNum * vParam.y);\n"
    "    float b = uWaveCoeffsB.x + uWaveCoeffsB.y * sin(uWaveNum * (vParam.x + vParam.y));\n"
    "    fragColor = vec3(r, g, b);\n"
    "}\n";

const char* fs_sin_uv_rgb = 
    "#version 330 core\n"
    "in vec2 vUV;\n"
    "uniform float uWaveNum;\n"
    "uniform vec3 uWaveCoeffsR;\n"
    "uniform vec3 uWaveCoeffsB;\n"
    "uniform vec3 uWaveCoeffsG;\n"
    "layout(location = 0) out vec3 fragColor;\n"
    "void main() {\n"
    "    float r = uWaveCoeffsR.x + uWaveCoeffsR.y * sin(uWaveNum * vUV.x);\n"
    "    float g = uWaveCoeffsG.x + uWaveCoeffsG.y * cos(uWaveNum * vUV.y);\n"
    "    float b = uWaveCoeffsB.x + uWaveCoeffsB.y * sin(uWaveNum * (vUV.x + vUV.y));\n"
    "    fragColor = vec3(r, g, b);\n"
    "}\n";

// Compute Shaders for resolving MSAA/single to SSBO
#if SSAA_SAMPLES > 1
#define TEXTURE_TARGET GL_TEXTURE_2D_MULTISAMPLE
#else
#define TEXTURE_TARGET GL_TEXTURE_2D
#endif

const char* cs_resolve_grey_single_src = 
    "#version 430 core\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "layout(binding = 0) uniform sampler2D single_image;\n"
    "layout(std430, binding = 1) buffer OutputBuffer {\n"
    "    float output_values[];\n"
    "};\n"
    "uniform int image_width;\n"
    "uniform int image_height;\n"
    "void main() {\n"
    "    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);\n"
    "    if (pixel.x >= image_width || pixel.y >= image_height) {\n"
    "        return;\n"
    "    }\n"
    "    float val = texelFetch(single_image, pixel, 0).r;\n"
    "    int output_y = image_height - 1 - pixel.y;\n"
    "    int output_index = output_y * image_width + pixel.x;\n"
    "    output_values[output_index] = val;\n"
    "}\n";

const char* cs_resolve_rgb_single_src = 
    "#version 430 core\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "layout(binding = 0) uniform sampler2D single_image;\n"
    "layout(std430, binding = 1) buffer OutputBuffer {\n"
    "    float output_values[];\n"
    "};\n"
    "uniform int image_width;\n"
    "uniform int image_height;\n"
    "void main() {\n"
    "    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);\n"
    "    if (pixel.x >= image_width || pixel.y >= image_height) {\n"
    "        return;\n"
    "    }\n"
    "    vec3 val = texelFetch(single_image, pixel, 0).rgb;\n"
    "    int output_y = image_height - 1 - pixel.y;\n"
    "    int pixel_index = output_y * image_width + pixel.x;\n"
    "    int output_base = 3 * pixel_index;\n"
    "    output_values[output_base + 0] = val.r;\n"
    "    output_values[output_base + 1] = val.g;\n"
    "    output_values[output_base + 2] = val.b;\n"
    "}\n";

const char* cs_resolve_grey_src = 
    "#version 430 core\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "layout(binding = 0) uniform sampler2DMS multisample_image;\n"
    "layout(std430, binding = 1) buffer OutputBuffer {\n"
    "    float output_values[];\n"
    "};\n"
    "uniform int image_width;\n"
    "uniform int image_height;\n"
    "uniform int sample_count;\n"
    "void main() {\n"
    "    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);\n"
    "    if (pixel.x >= image_width || pixel.y >= image_height) {\n"
    "        return;\n"
    "    }\n"
    "    float sample_sum = 0.0;\n"
    "    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {\n"
    "        sample_sum += texelFetch(multisample_image, pixel, sample_index).r;\n"
    "    }\n"
    "    int output_y = image_height - 1 - pixel.y;\n"
    "    int output_index = output_y * image_width + pixel.x;\n"
    "    output_values[output_index] = sample_sum / float(sample_count);\n"
    "}\n";

const char* cs_resolve_rgb_src = 
    "#version 430 core\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "layout(binding = 0) uniform sampler2DMS multisample_image;\n"
    "layout(std430, binding = 1) buffer OutputBuffer {\n"
    "    float output_values[];\n"
    "};\n"
    "uniform int image_width;\n"
    "uniform int image_height;\n"
    "uniform int sample_count;\n"
    "void main() {\n"
    "    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);\n"
    "    if (pixel.x >= image_width || pixel.y >= image_height) {\n"
    "        return;\n"
    "    }\n"
    "    vec3 sample_sum = vec3(0.0);\n"
    "    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {\n"
    "        sample_sum += texelFetch(multisample_image, pixel, sample_index).rgb;\n"
    "    }\n"
    "    vec3 resolved = sample_sum / float(sample_count);\n"
    "    int output_y = image_height - 1 - pixel.y;\n"
    "    int pixel_index = output_y * image_width + pixel.x;\n"
    "    int output_base = 3 * pixel_index;\n"
    "    output_values[output_base + 0] = resolved.r;\n"
    "    output_values[output_base + 1] = resolved.g;\n"
    "    output_values[output_base + 2] = resolved.b;\n"
    "}\n";

void save_bmp(const char* filename, float* data, int width, int height, int channels, int is_tex) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open output BMP: %s\n", filename);
        return;
    }

    int row_size = (width * 3 + 3) & ~3;
    int pixel_data_size = row_size * height;
    int file_size = 54 + pixel_data_size;

    unsigned char header[54] = {
        'B', 'M',
        file_size & 0xff, (file_size >> 8) & 0xff, (file_size >> 16) & 0xff, (file_size >> 24) & 0xff,
        0, 0, 0, 0,
        54, 0, 0, 0,
        40, 0, 0, 0,
        width & 0xff, (width >> 8) & 0xff, (width >> 16) & 0xff, (width >> 24) & 0xff,
        height & 0xff, (height >> 8) & 0xff, (height >> 16) & 0xff, (height >> 24) & 0xff,
        1, 0,
        24, 0,
        0, 0, 0, 0,
        pixel_data_size & 0xff, (pixel_data_size >> 8) & 0xff, (pixel_data_size >> 16) & 0xff, (pixel_data_size >> 24) & 0xff,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    fwrite(header, 1, 54, f);

    unsigned char* row_buf = calloc(1, row_size);
    for (int y = 0; y < height; ++y) {
        int flip_y = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            float r_val = 0.0f, g_val = 0.0f, b_val = 0.0f;
            if (channels == 3) {
                r_val = data[(flip_y * width + x) * 3 + 0];
                g_val = data[(flip_y * width + x) * 3 + 1];
                b_val = data[(flip_y * width + x) * 3 + 2];
            } else {
                r_val = data[flip_y * width + x];
                g_val = r_val;
                b_val = r_val;
            }

            float scale = is_tex ? 1.0f : 255.0f;
            int r = (int)(r_val * scale + 0.5f);
            int g = (int)(g_val * scale + 0.5f);
            int b = (int)(b_val * scale + 0.5f);

            r = r < 0 ? 0 : (r > 255 ? 255 : r);
            g = g < 0 ? 0 : (g > 255 ? 255 : g);
            b = b < 0 ? 0 : (b > 255 ? 255 : b);

            row_buf[x * 3 + 0] = (unsigned char)b;
            row_buf[x * 3 + 1] = (unsigned char)g;
            row_buf[x * 3 + 2] = (unsigned char)r;
        }
        fwrite(row_buf, 1, row_size, f);
    }

    free(row_buf);
    fclose(f);
}

// Write a specific CSV file
void write_csv(const char* path, CaseResult* results, int n_cases, int metric_type) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Failed to open CSV for writing: %s\n", path);
        return;
    }

    fprintf(f, "Case,Element,Shader,Interpolator,Total Elems,Vis Elems,Total Px,Shaded Px,"
               "Clear Time [ms],Draw Time [ms],Resolve Time [ms],"
               "Geom Time [ms],Raster Time [ms],Save Time [ms],Frame Time [ms],E2E Time [ms],"
               "Geom TP [MElem/s],Raster TP [MPx/s],Frame TP [MPx/s],E2E TP [MPx/s]\n");

    for (int i = 0; i < n_cases; ++i) {
        CaseResult r = results[i];
        double clear_val = 0.0;
        double draw_val = 0.0;
        double resolve_val = 0.0;
        double total_val = 0.0;
        switch (metric_type) {
            case 0:
                clear_val = r.clear_time_ms.median;
                draw_val = r.draw_time_ms.median;
                resolve_val = r.resolve_time_ms.median;
                total_val = r.time_ms.median;
                break;
            case 1:
                clear_val = r.clear_time_ms.min;
                draw_val = r.draw_time_ms.min;
                resolve_val = r.resolve_time_ms.min;
                total_val = r.time_ms.min;
                break;
            case 2:
                clear_val = r.clear_time_ms.max;
                draw_val = r.draw_time_ms.max;
                resolve_val = r.resolve_time_ms.max;
                total_val = r.time_ms.max;
                break;
            case 3:
                clear_val = r.clear_time_ms.mad;
                draw_val = r.draw_time_ms.mad;
                resolve_val = r.resolve_time_ms.mad;
                total_val = r.time_ms.mad;
                break;
            case 4:
                clear_val = r.clear_time_ms.cov;
                draw_val = r.draw_time_ms.cov;
                resolve_val = r.resolve_time_ms.cov;
                total_val = r.time_ms.cov;
                break;
        }

        double geom_time = 0.0;
        double raster_time = total_val;
        double save_time = 0.0;
        double frame_time = total_val;
        double e2e_time = total_val;

        double total_px = (double)(VIEW_WIDTH * VIEW_HEIGHT);
        double raster_tp = raster_time > 0.0 ? (total_px / (raster_time / 1000.0 * 1e6)) : 0.0;
        double frame_tp = frame_time > 0.0 ? (total_px / (frame_time / 1000.0 * 1e6)) : 0.0;
        double e2e_tp = e2e_time > 0.0 ? (total_px / (e2e_time / 1000.0 * 1e6)) : 0.0;

        if (metric_type == 4) {
            raster_tp = total_val;
            frame_tp = total_val;
            e2e_tp = total_val;
        }

        fprintf(f, "%s,%s,%s,%s,2.000000,2.000000,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,0.000000,%.6f,%.6f,%.6f\n",
                r.case_name, r.element, r.shader, r.interpolator,
                total_px, total_px, clear_val, draw_val, resolve_val,
                geom_time, raster_time, save_time, frame_time, e2e_time,
                raster_tp, frame_tp, e2e_tp);
    }

    fclose(f);
}

// Helper to parse CSV files
int load_csv_data(const char* base_dir, Vertex** vertices_out, int* num_vertices_out) {
    char path_coords[512];
    char path_connect[512];
    char path_field[512];
    char path_uvs[512];
    sprintf(path_coords, "%s/coords.csv", base_dir);
    sprintf(path_connect, "%s/connect.csv", base_dir);
    sprintf(path_field, "%s/field.csv", base_dir);
    sprintf(path_uvs, "%s/uvs.csv", base_dir);

    // Read coords
    FILE* f_coords = fopen(path_coords, "r");
    if (!f_coords) {
        fprintf(stderr, "Failed to open: %s\n", path_coords);
        return 0;
    }
    float coords[100][3];
    int num_coords = 0;
    while (num_coords < 100 && fscanf(f_coords, "%f,%f,%f\n", &coords[num_coords][0], &coords[num_coords][1], &coords[num_coords][2]) == 3) {
        num_coords++;
    }
    fclose(f_coords);

    // Read uvs
    FILE* f_uvs = fopen(path_uvs, "r");
    if (!f_uvs) {
        fprintf(stderr, "Failed to open: %s\n", path_uvs);
        return 0;
    }
    float uvs[100][2];
    int num_uvs = 0;
    while (num_uvs < 100 && fscanf(f_uvs, "%f,%f\n", &uvs[num_uvs][0], &uvs[num_uvs][1]) == 2) {
        num_uvs++;
    }
    fclose(f_uvs);

    // Read fields
    FILE* f_field = fopen(path_field, "r");
    if (!f_field) {
        fprintf(stderr, "Failed to open: %s\n", path_field);
        return 0;
    }
    float field_rgb[100][3];
    int num_fields = 0;
    while (num_fields < 100 && fscanf(f_field, "%f,%f,%f\n", &field_rgb[num_fields][0], &field_rgb[num_fields][1], &field_rgb[num_fields][2]) == 3) {
        num_fields++;
    }
    fclose(f_field);

    // Read connections and build triangles
    FILE* f_connect = fopen(path_connect, "r");
    if (!f_connect) {
        fprintf(stderr, "Failed to open: %s\n", path_connect);
        return 0;
    }
    int connect[100][3];
    int num_triangles = 0;
    while (num_triangles < 100 && fscanf(f_connect, "%d,%d,%d\n", &connect[num_triangles][0], &connect[num_triangles][1], &connect[num_triangles][2]) == 3) {
        num_triangles++;
    }
    fclose(f_connect);

    // Build the vertex array
    int num_vertices = num_triangles * 3;
    Vertex* vertices = malloc(num_vertices * sizeof(Vertex));
    for (int t = 0; t < num_triangles; ++t) {
        for (int v = 0; v < 3; ++v) {
            int idx = connect[t][v];
            vertices[t * 3 + v].pos[0] = coords[idx][0];
            vertices[t * 3 + v].pos[1] = coords[idx][1];
            vertices[t * 3 + v].pos[2] = coords[idx][2];
            vertices[t * 3 + v].uv[0] = uvs[idx][0];
            vertices[t * 3 + v].uv[1] = uvs[idx][1];
            
            // local parametric coordinates
            if (v == 0) {
                vertices[t * 3 + v].param[0] = 0.0f;
                vertices[t * 3 + v].param[1] = 0.0f;
            } else if (v == 1) {
                vertices[t * 3 + v].param[0] = 1.0f;
                vertices[t * 3 + v].param[1] = 0.0f;
            } else {
                vertices[t * 3 + v].param[0] = 0.0f;
                vertices[t * 3 + v].param[1] = 1.0f;
            }
            
            vertices[t * 3 + v].field_g = 0.299f * field_rgb[idx][0] + 0.587f * field_rgb[idx][1] + 0.114f * field_rgb[idx][2];
            vertices[t * 3 + v].field_rgb[0] = field_rgb[idx][0];
            vertices[t * 3 + v].field_rgb[1] = field_rgb[idx][1];
            vertices[t * 3 + v].field_rgb[2] = field_rgb[idx][2];
        }
    }

    *vertices_out = vertices;
    *num_vertices_out = num_vertices;
    return 1;
}

void compute_camera_params(const Vertex* vertices, int num_vertices,
                           float focal_length, const float pixels_size[2], const int pixels_num[2],
                           float pos_world[3], float roi_cent_world[3], float* image_dist_out) {
    float min_x = vertices[0].pos[0];
    float max_x = vertices[0].pos[0];
    float min_y = vertices[0].pos[1];
    float max_y = vertices[0].pos[1];
    float min_z = vertices[0].pos[2];
    float max_z = vertices[0].pos[2];
    for (int i = 1; i < num_vertices; ++i) {
        if (vertices[i].pos[0] < min_x) min_x = vertices[i].pos[0];
        if (vertices[i].pos[0] > max_x) max_x = vertices[i].pos[0];
        if (vertices[i].pos[1] < min_y) min_y = vertices[i].pos[1];
        if (vertices[i].pos[1] > max_y) max_y = vertices[i].pos[1];
        if (vertices[i].pos[2] < min_z) min_z = vertices[i].pos[2];
        if (vertices[i].pos[2] > max_z) max_z = vertices[i].pos[2];
    }
    
    roi_cent_world[0] = 0.5f * (min_x + max_x);
    roi_cent_world[1] = 0.5f * (min_y + max_y);
    roi_cent_world[2] = 0.5f * (min_z + max_z);
    
    float max_abs_x = 0.5f * (max_x - min_x);
    float max_abs_y = 0.5f * (max_y - min_y);
    
    float fov_leng_x = 2.0f * max_abs_x;
    float fov_leng_y = 2.0f * max_abs_y;
    
    float sensor_size_x = (float)pixels_num[0] * pixels_size[0];
    float sensor_size_y = (float)pixels_num[1] * pixels_size[1];
    
    float dist_x = (fov_leng_x * focal_length) / sensor_size_x;
    float dist_y = (fov_leng_y * focal_length) / sensor_size_y;
    float image_dist = dist_x > dist_y ? dist_x : dist_y;
    *image_dist_out = image_dist;
    
    pos_world[0] = roi_cent_world[0];
    pos_world[1] = roi_cent_world[1];
    pos_world[2] = roi_cent_world[2] + image_dist;
}

void compute_camera_matrices(const float pos_world[3], const float roi_cent_world[3],
                             float focal_length, const float pixels_size[2], const int pixels_num[2],
                             float view_matrix[16], float proj_matrix[16]) {
    (void)roi_cent_world;
    view_matrix[0] = 1.0f;  view_matrix[1] = 0.0f;  view_matrix[2] = 0.0f;  view_matrix[3] = 0.0f;
    view_matrix[4] = 0.0f;  view_matrix[5] = 1.0f;  view_matrix[6] = 0.0f;  view_matrix[7] = 0.0f;
    view_matrix[8] = 0.0f;  view_matrix[9] = 0.0f;  view_matrix[10] = 1.0f; view_matrix[11] = 0.0f;
    view_matrix[12] = -pos_world[0];
    view_matrix[13] = -pos_world[1];
    view_matrix[14] = -pos_world[2];
    view_matrix[15] = 1.0f;

    float fx = focal_length / pixels_size[0];
    float fy = focal_length / pixels_size[1];
    
    float near_plane = 0.1f;
    float far_plane = 1000.0f;
    
    proj_matrix[0] = (2.0f * fx) / (float)pixels_num[0];
    proj_matrix[1] = 0.0f;
    proj_matrix[2] = 0.0f;
    proj_matrix[3] = 0.0f;
    
    proj_matrix[4] = 0.0f;
    proj_matrix[5] = (2.0f * fy) / (float)pixels_num[1];
    proj_matrix[6] = 0.0f;
    proj_matrix[7] = 0.0f;
    
    proj_matrix[8] = 0.0f;
    proj_matrix[9] = 0.0f;
    proj_matrix[10] = -(far_plane + near_plane) / (far_plane - near_plane);
    proj_matrix[11] = -1.0f;
    
    proj_matrix[12] = 0.0f;
    proj_matrix[13] = 0.0f;
    proj_matrix[14] = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
    proj_matrix[15] = 0.0f;
}

int main() {
    force_llvmpipe_single_thread();
    system("mkdir -p temp");
    char run_timestamp[32];
    make_timestamp_string(run_timestamp, sizeof(run_timestamp));
    rename_if_exists("temp/out", "temp/out_ssaa1");
    {
        char out_dir_cmd[128];
        snprintf(out_dir_cmd, sizeof(out_dir_cmd), "mkdir -p temp/out_%s",
                 OUT_TAG);
        system(out_dir_cmd);
    }
    rename_if_exists(
        "temp/llvmpipe_stats_median.csv",
        "temp/llvmpipe_stats_median_ssaa1.csv"
    );
    rename_if_exists(
        "temp/llvmpipe_stats_min.csv",
        "temp/llvmpipe_stats_min_ssaa1.csv"
    );
    rename_if_exists(
        "temp/llvmpipe_stats_max.csv",
        "temp/llvmpipe_stats_max_ssaa1.csv"
    );
    rename_if_exists(
        "temp/llvmpipe_stats_mad.csv",
        "temp/llvmpipe_stats_mad_ssaa1.csv"
    );
    rename_if_exists(
        "temp/llvmpipe_stats_cov.csv",
        "temp/llvmpipe_stats_cov_ssaa1.csv"
    );

    // 1. Initialize Headless EGL
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get default EGL display.\n");
        return 1;
    }

    EGLint major, minor;
    if (!eglInitialize(display, &major, &minor)) {
        fprintf(stderr, "Failed to initialize EGL.\n");
        return 1;
    }

    eglBindAPI(EGL_OPENGL_API);

    EGLConfig config;
    EGLint num_configs;
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "Failed to choose EGL config.\n");
        return 1;
    }

    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Failed to create EGL context.\n");
        return 1;
    }

    EGLint pbuffer_attribs[] = {
        EGL_WIDTH, 1,
        EGL_HEIGHT, 1,
        EGL_NONE
    };
    EGLSurface surface = eglCreatePbufferSurface(display, config, pbuffer_attribs);
    if (surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Failed to create pbuffer surface.\n");
        return 1;
    }

    if (!eglMakeCurrent(display, surface, surface, context)) {
        fprintf(stderr, "Failed to make EGL context current.\n");
        return 1;
    }

    printf("OpenGL context initialized. Renderer: %s\n", glGetString(GL_RENDERER));

    // Set the viewport size to match target resolution
    glViewport(0, 0, VIEW_WIDTH, VIEW_HEIGHT);

    // 2. Load textures
    int tex_w = 0, tex_h = 0;
    float* tex_data_g = load_bmp("texture/speckle.bmp", &tex_w, &tex_h, 1);
    if (!tex_data_g) {
        fprintf(stderr, "Could not load texture/speckle.bmp\n");
        return 1;
    }

    int tex_w_rgb = 0, tex_h_rgb = 0;
    float* tex_data_rgb = load_bmp("texture/speckle_rgb.bmp", &tex_w_rgb, &tex_h_rgb, 3);
    if (!tex_data_rgb) {
        fprintf(stderr, "Could not load texture/speckle_rgb.bmp\n");
        free(tex_data_g);
        return 1;
    }

    printf("Textures loaded. Speckle (grey): %dx%d, Speckle (rgb): %dx%d\n",
           tex_w, tex_h, tex_w_rgb, tex_h_rgb);

    // 3. Upload textures to GPU (software memory)
    GLuint gl_tex_g;
    glGenTextures(1, &gl_tex_g);
    glBindTexture(GL_TEXTURE_2D, gl_tex_g);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, tex_w, tex_h, 0, GL_RED, GL_FLOAT, tex_data_g);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint gl_tex_rgb;
    glGenTextures(1, &gl_tex_rgb);
    glBindTexture(GL_TEXTURE_2D, gl_tex_rgb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, tex_w_rgb, tex_h_rgb, 0, GL_RGB, GL_FLOAT, tex_data_rgb);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    free(tex_data_g);
    free(tex_data_rgb);

    // 4. Create render targets as MSAA textures
    GLuint fbo_msaa = 0;
    GLuint colour_texture_grey = 0;
    GLuint colour_texture_rgb = 0;
    GLint actual_samples = 1;

    glGenFramebuffers(1, &fbo_msaa);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_msaa);

    glGenTextures(1, &colour_texture_grey);
    glBindTexture(TEXTURE_TARGET, colour_texture_grey);
#if SSAA_SAMPLES > 1
    glTexImage2DMultisample(
        GL_TEXTURE_2D_MULTISAMPLE,
        SSAA_SAMPLES,
        GL_R32F,
        VIEW_WIDTH,
        VIEW_HEIGHT,
        GL_TRUE
    );
#else
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R32F,
        VIEW_WIDTH,
        VIEW_HEIGHT,
        0,
        GL_RED,
        GL_FLOAT,
        NULL
    );
#endif

    glGenTextures(1, &colour_texture_rgb);
    glBindTexture(TEXTURE_TARGET, colour_texture_rgb);
#if SSAA_SAMPLES > 1
    glTexImage2DMultisample(
        GL_TEXTURE_2D_MULTISAMPLE,
        SSAA_SAMPLES,
        GL_RGBA32F,
        VIEW_WIDTH,
        VIEW_HEIGHT,
        GL_TRUE
    );
#else
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA32F,
        VIEW_WIDTH,
        VIEW_HEIGHT,
        0,
        GL_RGBA,
        GL_FLOAT,
        NULL
    );
#endif

    // Query actual supported samples
#if SSAA_SAMPLES > 1
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, colour_texture_grey);
    glGetTexLevelParameteriv(
        GL_TEXTURE_2D_MULTISAMPLE,
        0,
        GL_TEXTURE_SAMPLES,
        &actual_samples
    );
#else
    actual_samples = 1;
#endif

    // Allocate persistently mapped SSBO once (large enough for RGB)
    GLuint output_ssbo = 0;
    size_t output_size_bytes = (size_t)VIEW_WIDTH * (size_t)VIEW_HEIGHT * 3 * sizeof(float);
    glGenBuffers(1, &output_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_ssbo);
    glBufferStorage(
        GL_SHADER_STORAGE_BUFFER,
        output_size_bytes,
        NULL,
        GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT
    );
    float* output_pixels = (float*)glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER,
        0,
        output_size_bytes,
        GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT
    );
    if (!output_pixels) {
        fprintf(stderr, "Failed to map SSBO.\n");
        return 1;
    }

    if (SSAA_SAMPLES > 1) {
        glEnable(GL_MULTISAMPLE);
        glEnable(GL_SAMPLE_SHADING);
        glMinSampleShading(1.0f);
    }
    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    printf(
        "Render setup: requested_samples=%d actual_samples=%d out_tag=%s\n",
        SSAA_SAMPLES,
        actual_samples,
        OUT_TAG
    );

    // 5. Geometry layout setup (load from tilted dataset coords)
    Vertex* vertices = NULL;
    int num_vertices = 0;
    if (!load_csv_data("data/tilt/tri3_fullraster", &vertices, &num_vertices)) {
        fprintf(stderr, "Failed to load CSV data.\n");
        return 1;
    }

    float focal_length = 50.0e-3f;
    float pixels_size[2] = { 5.3e-6f, 5.3e-6f };
    int pixels_num[2] = { VIEW_WIDTH, VIEW_HEIGHT };
    float pos_world[3];
    float roi_cent_world[3];
    float image_dist;
    compute_camera_params(vertices, num_vertices, focal_length, pixels_size, pixels_num, pos_world, roi_cent_world, &image_dist);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(Vertex), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, param));

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, field_g));

    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, field_rgb));

    // 6. Shader compilation & program linking
    GLuint vs_n_g = compile_shader(GL_VERTEX_SHADER, vs_nodal_grey);
    GLuint fs_n_g = compile_shader(GL_FRAGMENT_SHADER, fs_nodal_grey);
    GLuint prog_nodal_grey = link_program(vs_n_g, fs_n_g);

    GLuint vs_n_r = compile_shader(GL_VERTEX_SHADER, vs_nodal_rgb);
    GLuint fs_n_r = compile_shader(GL_FRAGMENT_SHADER, fs_nodal_rgb);
    GLuint prog_nodal_rgb = link_program(vs_n_r, fs_n_r);

    GLuint vs_u = compile_shader(GL_VERTEX_SHADER, vs_uv);
    GLuint fs_t_g_l = compile_shader(GL_FRAGMENT_SHADER, fs_tex8_grey_linear);
    GLuint prog_tex8_grey_linear = link_program(vs_u, fs_t_g_l);

    GLuint fs_t_r_l = compile_shader(GL_FRAGMENT_SHADER, fs_tex8_rgb_linear);
    GLuint prog_tex8_rgb_linear = link_program(vs_u, fs_t_r_l);

    GLuint fs_t_g_c = compile_shader(GL_FRAGMENT_SHADER, fs_tex8_grey_cubic);
    GLuint prog_tex8_grey_cubic = link_program(vs_u, fs_t_g_c);

    GLuint fs_t_r_c = compile_shader(GL_FRAGMENT_SHADER, fs_tex8_rgb_cubic);
    GLuint prog_tex8_rgb_cubic = link_program(vs_u, fs_t_r_c);

    GLuint fs_t_g_q = compile_shader(GL_FRAGMENT_SHADER, fs_tex8_grey_quintic);
    GLuint prog_tex8_grey_quintic = link_program(vs_u, fs_t_g_q);

    GLuint fs_t_r_q = compile_shader(GL_FRAGMENT_SHADER, fs_tex8_rgb_quintic);
    GLuint prog_tex8_rgb_quintic = link_program(vs_u, fs_t_r_q);

    GLuint vs_p = compile_shader(GL_VERTEX_SHADER, vs_param);

    GLuint fs_c_g = compile_shader(GL_FRAGMENT_SHADER, fs_const_grey);
    GLuint prog_const_grey = link_program(vs_p, fs_c_g);
    GLuint prog_const_grey_uv = link_program(vs_u, fs_c_g);

    GLuint fs_c_r = compile_shader(GL_FRAGMENT_SHADER, fs_const_rgb);
    GLuint prog_const_rgb = link_program(vs_p, fs_c_r);
    GLuint prog_const_rgb_uv = link_program(vs_u, fs_c_r);

    GLuint fs_l_p_g = compile_shader(GL_FRAGMENT_SHADER, fs_linear_param_grey);
    GLuint prog_linear_param_grey = link_program(vs_p, fs_l_p_g);

    GLuint fs_l_u_g = compile_shader(GL_FRAGMENT_SHADER, fs_linear_uv_grey);
    GLuint prog_linear_uv_grey = link_program(vs_u, fs_l_u_g);

    GLuint fs_l_p_r = compile_shader(GL_FRAGMENT_SHADER, fs_linear_param_rgb);
    GLuint prog_linear_param_rgb = link_program(vs_p, fs_l_p_r);

    GLuint fs_l_u_r = compile_shader(GL_FRAGMENT_SHADER, fs_linear_uv_rgb);
    GLuint prog_linear_uv_rgb = link_program(vs_u, fs_l_u_r);

    GLuint fs_q_p_g = compile_shader(
        GL_FRAGMENT_SHADER,
        fs_quadratic_param_grey
    );
    GLuint prog_quadratic_param_grey = link_program(vs_p, fs_q_p_g);

    GLuint fs_q_u_g = compile_shader(
        GL_FRAGMENT_SHADER,
        fs_quadratic_uv_grey
    );
    GLuint prog_quadratic_uv_grey = link_program(vs_u, fs_q_u_g);

    GLuint fs_q_p_r = compile_shader(
        GL_FRAGMENT_SHADER,
        fs_quadratic_param_rgb
    );
    GLuint prog_quadratic_param_rgb = link_program(vs_p, fs_q_p_r);

    GLuint fs_q_u_r = compile_shader(
        GL_FRAGMENT_SHADER,
        fs_quadratic_uv_rgb
    );
    GLuint prog_quadratic_uv_rgb = link_program(vs_u, fs_q_u_r);

    GLuint fs_s_p_g = compile_shader(GL_FRAGMENT_SHADER, fs_sin_param_grey);
    GLuint prog_sin_param_grey = link_program(vs_p, fs_s_p_g);

    GLuint fs_s_u_g = compile_shader(GL_FRAGMENT_SHADER, fs_sin_uv_grey);
    GLuint prog_sin_uv_grey = link_program(vs_u, fs_s_u_g);

    GLuint fs_s_p_r = compile_shader(GL_FRAGMENT_SHADER, fs_sin_param_rgb);
    GLuint prog_sin_param_rgb = link_program(vs_p, fs_s_p_r);

    GLuint fs_s_u_r = compile_shader(GL_FRAGMENT_SHADER, fs_sin_uv_rgb);
    GLuint prog_sin_uv_rgb = link_program(vs_u, fs_s_u_r);

    // Resolve Compute Shaders
#if SSAA_SAMPLES > 1
    GLuint cs_grey = compile_shader(GL_COMPUTE_SHADER, cs_resolve_grey_src);
#else
    GLuint cs_grey = compile_shader(GL_COMPUTE_SHADER, cs_resolve_grey_single_src);
#endif
    GLuint prog_resolve_grey = glCreateProgram();
    glAttachShader(prog_resolve_grey, cs_grey);
    glLinkProgram(prog_resolve_grey);

#if SSAA_SAMPLES > 1
    GLuint cs_rgb = compile_shader(GL_COMPUTE_SHADER, cs_resolve_rgb_src);
#else
    GLuint cs_rgb = compile_shader(GL_COMPUTE_SHADER, cs_resolve_rgb_single_src);
#endif
    GLuint prog_resolve_rgb = glCreateProgram();
    glAttachShader(prog_resolve_rgb, cs_rgb);
    glLinkProgram(prog_resolve_rgb);

    // 7. Define benchmark cases
    typedef struct {
        char case_name[64];
        char element[16];
        char shader[32];
        char interpolator[32];
        GLuint program;
        int is_rgb;
        GLuint texture;
        int has_tex_size;
        float tex_w;
        float tex_h;
        float wave_num;
    } CaseSpec;

    float wave_param = (float)(2.0 * PI * 2.0);
    float wave_uv = (float)(2.0 * PI * 6.0);

    CaseSpec cases[] = {
        { "tri3_nodal_grey", "tri3", "nodal_grey", "nodal", prog_nodal_grey, 0, 0, 0, 0, 0, 0.0f },
        { "tri3_nodal_rgb", "tri3", "nodal_rgb", "nodal", prog_nodal_rgb, 1, 0, 0, 0, 0, 0.0f },
        { "tri3_tex8_grey_linear_direct", "tri3", "tex8_grey", "linear", prog_tex8_grey_linear, 0, gl_tex_g, 0, 0, 0, 0.0f },
        { "tri3_tex8_grey_cubic_catmull_rom_direct", "tri3", "tex8_grey", "cubic_catmull_rom", prog_tex8_grey_cubic, 0, gl_tex_g, 1, (float)tex_w, (float)tex_h, 0.0f },
        { "tri3_tex8_grey_quintic_bspline_direct", "tri3", "tex8_grey", "quintic_bspline", prog_tex8_grey_quintic, 0, gl_tex_g, 1, (float)tex_w, (float)tex_h, 0.0f },
        { "tri3_tex8_rgb_linear_direct", "tri3", "tex8_rgb", "linear", prog_tex8_rgb_linear, 1, gl_tex_rgb, 0, 0, 0, 0.0f },
        { "tri3_tex8_rgb_cubic_catmull_rom_direct", "tri3", "tex8_rgb", "cubic_catmull_rom", prog_tex8_rgb_cubic, 1, gl_tex_rgb, 1, (float)tex_w_rgb, (float)tex_h_rgb, 0.0f },
        { "tri3_tex8_rgb_quintic_bspline_direct", "tri3", "tex8_rgb", "quintic_bspline", prog_tex8_rgb_quintic, 1, gl_tex_rgb, 1, (float)tex_w_rgb, (float)tex_h_rgb, 0.0f },
        { "tri3_texfunc_grey_param_constant", "tri3", "texfunc_grey", "constant", prog_const_grey, 0, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_grey_uv_constant", "tri3", "texfunc_grey", "constant", prog_const_grey_uv, 0, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_grey_param_linear", "tri3", "texfunc_grey", "linear", prog_linear_param_grey, 0, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_grey_uv_linear", "tri3", "texfunc_grey", "linear", prog_linear_uv_grey, 0, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_grey_param_quadratic", "tri3", "texfunc_grey", "quadratic", prog_quadratic_param_grey, 0, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_grey_uv_quadratic", "tri3", "texfunc_grey", "quadratic", prog_quadratic_uv_grey, 0, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_grey_param_sinusoidal", "tri3", "texfunc_grey", "sinusoidal", prog_sin_param_grey, 0, 0, 0, 0, 0, wave_param },
        { "tri3_texfunc_grey_uv_sinusoidal", "tri3", "texfunc_grey", "sinusoidal", prog_sin_uv_grey, 0, 0, 0, 0, 0, wave_uv },
        { "tri3_texfunc_rgb_param_constant", "tri3", "texfunc_rgb", "constant", prog_const_rgb, 1, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_rgb_uv_constant", "tri3", "texfunc_rgb", "constant", prog_const_rgb_uv, 1, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_rgb_param_linear", "tri3", "texfunc_rgb", "linear", prog_linear_param_rgb, 1, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_rgb_uv_linear", "tri3", "texfunc_rgb", "linear", prog_linear_uv_rgb, 1, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_rgb_param_quadratic", "tri3", "texfunc_rgb", "quadratic", prog_quadratic_param_rgb, 1, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_rgb_uv_quadratic", "tri3", "texfunc_rgb", "quadratic", prog_quadratic_uv_rgb, 1, 0, 0, 0, 0, 0.0f },
        { "tri3_texfunc_rgb_param_sinusoidal", "tri3", "texfunc_rgb", "sinusoidal", prog_sin_param_rgb, 1, 0, 0, 0, 0, wave_param },
        { "tri3_texfunc_rgb_uv_sinusoidal", "tri3", "texfunc_rgb", "sinusoidal", prog_sin_uv_rgb, 1, 0, 0, 0, 0, wave_uv }
    };

    int num_cases = sizeof(cases) / sizeof(CaseSpec);
    CaseResult* case_results = malloc(num_cases * sizeof(CaseResult));

    printf("Starting LLVMpipe benchmarking of %d cases...\n", num_cases);

    GLint loc_width_grey = glGetUniformLocation(prog_resolve_grey, "image_width");
    GLint loc_height_grey = glGetUniformLocation(prog_resolve_grey, "image_height");
    GLint loc_samples_grey = glGetUniformLocation(prog_resolve_grey, "sample_count");

    GLint loc_width_rgb = glGetUniformLocation(prog_resolve_rgb, "image_width");
    GLint loc_height_rgb = glGetUniformLocation(prog_resolve_rgb, "image_height");
    GLint loc_samples_rgb = glGetUniformLocation(prog_resolve_rgb, "sample_count");

    int resolve_groups_x = (VIEW_WIDTH + 15) / 16;
    int resolve_groups_y = (VIEW_HEIGHT + 15) / 16;

    for (int c = 0; c < num_cases; ++c) {
        CaseSpec spec = cases[c];
        int channels = spec.is_rgb ? 3 : 1;
        printf("Benchmarking: %s...\n", spec.case_name);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_msaa);
        if (spec.is_rgb) {
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                TEXTURE_TARGET,
                colour_texture_rgb,
                0
            );
        } else {
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                TEXTURE_TARGET,
                colour_texture_grey,
                0
            );
        }

        glUseProgram(spec.program);

        // Bind texture if needed
        if (spec.texture != 0) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, spec.texture);
            GLint tex_loc = glGetUniformLocation(spec.program, "uTexture");
            if (tex_loc != -1) glUniform1i(tex_loc, 0);

            if (spec.has_tex_size) {
                GLint size_loc = glGetUniformLocation(spec.program, "uTextureSize");
                if (size_loc != -1) glUniform2f(size_loc, spec.tex_w, spec.tex_h);
            }
        }

        // Cache uniform locations outside the hot loop
        GLint view_loc = glGetUniformLocation(spec.program, "uViewMatrix");
        GLint proj_loc = glGetUniformLocation(spec.program, "uProjMatrix");

        // Calculate matrices outside the hot loop
        float view_matrix[16];
        float proj_matrix[16];
        compute_camera_matrices(pos_world, roi_cent_world, focal_length, pixels_size, pixels_num, view_matrix, proj_matrix);

        // Upload them outside the hot loop
        if (view_loc != -1) glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix);
        if (proj_loc != -1) glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj_matrix);

        // Upload constant shader parameters outside the hot loop
        GLint const_color_loc = glGetUniformLocation(spec.program, "uConstantColor");
        if (const_color_loc != -1) glUniform1f(const_color_loc, 0.5f);

        GLint const_color_rgb_loc = glGetUniformLocation(spec.program, "uConstantColorRGB");
        if (const_color_rgb_loc != -1) glUniform3f(const_color_rgb_loc, 0.2f, 0.5f, 0.8f);

        GLint linear_coeffs_loc = glGetUniformLocation(
            spec.program,
            "uLinearCoeffs[0]"
        );
        if (linear_coeffs_loc != -1) {
            glUniform1fv(linear_coeffs_loc, 3, RILEY_LINEAR_COEFFS);
        }

        GLint linear_coeffs_rgb_loc = glGetUniformLocation(
            spec.program,
            "uLinearCoeffsRGB[0]"
        );
        if (linear_coeffs_rgb_loc != -1) {
            glUniform1fv(
                linear_coeffs_rgb_loc,
                9,
                RILEY_LINEAR_COEFFS_RGB
            );
        }

        GLint quadratic_coeffs_loc = glGetUniformLocation(
            spec.program,
            "uQuadraticCoeffs[0]"
        );
        if (quadratic_coeffs_loc != -1) {
            glUniform1fv(
                quadratic_coeffs_loc,
                6,
                RILEY_QUADRATIC_COEFFS
            );
        }

        GLint quadratic_coeffs_rgb_loc = glGetUniformLocation(
            spec.program,
            "uQuadraticCoeffsRGB[0]"
        );
        if (quadratic_coeffs_rgb_loc != -1) {
            glUniform1fv(
                quadratic_coeffs_rgb_loc,
                18,
                RILEY_QUADRATIC_COEFFS_RGB
            );
        }

        GLint wave_coeffs_loc = glGetUniformLocation(spec.program, "uWaveCoeffs");
        if (wave_coeffs_loc != -1) glUniform3f(wave_coeffs_loc, 0.5f, 0.25f, 0.20f);

        GLint wave_coeffs_r_loc = glGetUniformLocation(spec.program, "uWaveCoeffsR");
        if (wave_coeffs_r_loc != -1) glUniform3f(wave_coeffs_r_loc, 0.5f, 0.25f, 0.0f);
        GLint wave_coeffs_g_loc = glGetUniformLocation(spec.program, "uWaveCoeffsG");
        if (wave_coeffs_g_loc != -1) glUniform3f(wave_coeffs_g_loc, 0.5f, 0.25f, 0.0f);
        GLint wave_coeffs_b_loc = glGetUniformLocation(spec.program, "uWaveCoeffsB");
        if (wave_coeffs_b_loc != -1) glUniform3f(wave_coeffs_b_loc, 0.5f, 0.20f, 0.0f);

        if (spec.wave_num != 0.0f) {
            GLint wave_loc = glGetUniformLocation(spec.program, "uWaveNum");
            if (wave_loc != -1) glUniform1f(wave_loc, spec.wave_num);
        }

        // Perform one warmup run
        if (spec.is_rgb) {
            glClearColor(0.2f, 0.5f, 0.8f, 1.0f);
        } else {
            glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        }
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, num_vertices);

        // Resolve
        if (spec.is_rgb) {
            glUseProgram(prog_resolve_rgb);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(TEXTURE_TARGET, colour_texture_rgb);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_ssbo);
            glUniform1i(loc_width_rgb, VIEW_WIDTH);
            glUniform1i(loc_height_rgb, VIEW_HEIGHT);
            if (loc_samples_rgb != -1) glUniform1i(loc_samples_rgb, SSAA_SAMPLES);
        } else {
            glUseProgram(prog_resolve_grey);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(TEXTURE_TARGET, colour_texture_grey);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_ssbo);
            glUniform1i(loc_width_grey, VIEW_WIDTH);
            glUniform1i(loc_height_grey, VIEW_HEIGHT);
            if (loc_samples_grey != -1) glUniform1i(loc_samples_grey, SSAA_SAMPLES);
        }
        glDispatchCompute(resolve_groups_x, resolve_groups_y, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        glFinish();

        // Save resolved output render from the mapped buffer
        char out_filename[256];
        sprintf(
            out_filename,
            "temp/out_%s/%s_%s_%s.bmp",
            OUT_TAG,
            spec.case_name,
            OUT_TAG,
            run_timestamp
        );
        int is_tex = (strstr(spec.case_name, "tex8") != NULL);
        save_bmp(
            out_filename,
            output_pixels,
            VIEW_WIDTH,
            VIEW_HEIGHT,
            channels,
            is_tex
        );

        // Benchmark runs
        double total_times[RUNS];
        for (int r = 0; r < RUNS; ++r) {
            glFinish();
            double start_e2e = get_time_ms();

            glUseProgram(spec.program);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawArrays(GL_TRIANGLES, 0, num_vertices);

            if (spec.is_rgb) {
                glUseProgram(prog_resolve_rgb);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(TEXTURE_TARGET, colour_texture_rgb);
            } else {
                glUseProgram(prog_resolve_grey);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(TEXTURE_TARGET, colour_texture_grey);
            }
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_ssbo);
            glDispatchCompute(resolve_groups_x, resolve_groups_y, 1);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
            glFinish();

            double end_e2e = get_time_ms();
            total_times[r] = end_e2e - start_e2e;
        }

        // Compute statistics
        MetricStats stats = calc_metric_stats(total_times, RUNS);
        printf("  Median E2E: %.2f ms, CoV: %.2f%%, Throughput: %.2f MPx/s\n",
               stats.median, stats.cov,
               (VIEW_WIDTH * VIEW_HEIGHT) / (stats.median / 1000.0 * 1e6));

        // Save results
        MetricStats zero_stats = {0.0, 0.0, 0.0, 0.0, 0.0};
        strcpy(case_results[c].case_name, spec.case_name);
        strcpy(case_results[c].element, spec.element);
        strcpy(case_results[c].shader, spec.shader);
        strcpy(case_results[c].interpolator, spec.interpolator);
        case_results[c].clear_time_ms = zero_stats;
        case_results[c].draw_time_ms = zero_stats;
        case_results[c].resolve_time_ms = zero_stats;
        case_results[c].time_ms = stats;
    }

    // 8. Output results to the 5 CSV files matching Riley format
    {
        char csv_path[128];
        snprintf(
            csv_path,
            sizeof(csv_path),
            "temp/llvmpipe_stats_median_%s_%s.csv",
            OUT_TAG,
            run_timestamp
        );
        write_csv(csv_path, case_results, num_cases, 0);
        snprintf(
            csv_path,
            sizeof(csv_path),
            "temp/llvmpipe_stats_min_%s_%s.csv",
            OUT_TAG,
            run_timestamp
        );
        write_csv(csv_path, case_results, num_cases, 1);
        snprintf(
            csv_path,
            sizeof(csv_path),
            "temp/llvmpipe_stats_max_%s_%s.csv",
            OUT_TAG,
            run_timestamp
        );
        write_csv(csv_path, case_results, num_cases, 2);
        snprintf(
            csv_path,
            sizeof(csv_path),
            "temp/llvmpipe_stats_mad_%s_%s.csv",
            OUT_TAG,
            run_timestamp
        );
        write_csv(csv_path, case_results, num_cases, 3);
        snprintf(
            csv_path,
            sizeof(csv_path),
            "temp/llvmpipe_stats_cov_%s_%s.csv",
            OUT_TAG,
            run_timestamp
        );
        write_csv(csv_path, case_results, num_cases, 4);
    }

    printf(
        "LLVMpipe benchmark completed successfully. "
        "CSV reports written to temp/llvmpipe_stats_*_%s_%s.csv\n",
        OUT_TAG,
        run_timestamp
    );

    // Clean up GL resources
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteTextures(1, &gl_tex_g);
    glDeleteTextures(1, &gl_tex_rgb);
    glDeleteTextures(1, &colour_texture_grey);
    glDeleteTextures(1, &colour_texture_rgb);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_ssbo);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glDeleteBuffers(1, &output_ssbo);
    glDeleteFramebuffers(1, &fbo_msaa);

    glDeleteProgram(prog_nodal_grey);
    glDeleteProgram(prog_nodal_rgb);
    glDeleteProgram(prog_tex8_grey_linear);
    glDeleteProgram(prog_tex8_rgb_linear);
    glDeleteProgram(prog_tex8_grey_cubic);
    glDeleteProgram(prog_tex8_rgb_cubic);
    glDeleteProgram(prog_const_grey);
    glDeleteProgram(prog_const_grey_uv);
    glDeleteProgram(prog_const_rgb);
    glDeleteProgram(prog_const_rgb_uv);
    glDeleteProgram(prog_linear_param_grey);
    glDeleteProgram(prog_linear_uv_grey);
    glDeleteProgram(prog_linear_param_rgb);
    glDeleteProgram(prog_linear_uv_rgb);
    glDeleteProgram(prog_quadratic_param_grey);
    glDeleteProgram(prog_quadratic_uv_grey);
    glDeleteProgram(prog_quadratic_param_rgb);
    glDeleteProgram(prog_quadratic_uv_rgb);
    glDeleteProgram(prog_sin_param_grey);
    glDeleteProgram(prog_sin_uv_grey);
    glDeleteProgram(prog_sin_param_rgb);
    glDeleteProgram(prog_sin_uv_rgb);
    glDeleteProgram(prog_resolve_grey);
    glDeleteProgram(prog_resolve_rgb);
    free(case_results);
    free(vertices);

    eglDestroySurface(display, surface);
    eglDestroyContext(display, context);
    eglTerminate(display);

    return 0;
}
