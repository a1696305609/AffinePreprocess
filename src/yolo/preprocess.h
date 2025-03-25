#ifndef __PREPROCESS_H
#define __PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>


struct AffineMatrix2{
    float v_x[3];
    float v_y[3];
};


void uint8_2_float(const uint8_t* src, float* dst, int channels, int height, int width);

void mat2tensor(float* src, float* dst, int channels, int src_height, int src_width, int dst_height, int dst_width, float padding_value);

void mat2tensor_crop(float* src, float* dst, int channels, int src_height, int src_width, int dst_height, int dst_width, int crop_size, float padding_value);

#endif  // __PREPROCESS_H
