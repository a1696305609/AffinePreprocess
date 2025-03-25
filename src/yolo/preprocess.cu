#include "preprocess.h"
#include <opencv2/opencv.hpp>


__global__ void uint8_2_float_kernel(const uint8_t* src, float* dst, int channels, int height, int width){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 计算像素的 x 坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 计算像素的 y 坐标
    if (x >= width || y >= height) return;

    for (int i = 0; i < channels; ++i){
        int index = (y * width + x)*channels + i;
        dst[index] = src[index] / 1.f;
    }

}

void uint8_2_float(const uint8_t* src, float* dst, int channels, int height, int width){
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x -1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    uint8_2_float_kernel<<<gridSize, blockSize>>>(src, dst, channels, height, width);
}

__global__ void mat2tensor_kernel(float* src, float* dst, int channels, int src_height, int src_width, int dst_height, int dst_width, float padding_value, AffineMatrix2 dst2src){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 计算像素的 x 坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 计算像素的 y 坐标
    if (x >= dst_width || y >= dst_height) return;

    int dst_num_pixels = dst_height * dst_width;

    float src_x = dst2src.v_x[0] * x + dst2src.v_x[2] + 0.5f;
    float src_y = dst2src.v_y[1] * y + dst2src.v_y[2] + 0.5f;
    
    int index = y * dst_width + x;
    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
        for (int i = 0; i< channels; ++i){
            dst[index + i * dst_num_pixels] = padding_value;
        }
        return;
    }
    int x_low = floorf(src_x);
    int y_low = floorf(src_y);
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    float lx = src_x - x_low;
    float hx = 1.f - lx;
    float ly = src_y - y_low;
    float hy = 1.f - ly;

    float w_lt = lx * ly, w_rt = hx * ly, w_lb = lx * hy, w_rb = hx * hy;

    float padding_array[] = {padding_value, padding_value, padding_value};
    float* v_lt = padding_array;
    float* v_rt = padding_array;
    float* v_lb = padding_array;
    float* v_rb = padding_array;
    if (y_low >= 0){
        if (x_low >= 0){
            v_lt = src + (y_low * src_width  + x_low) * 3;
        }
        if (x_high < src_width){
            v_rt = src + (y_low * src_width + x_high) * 3;
        }
    }
    if (y_high < src_height){
        if (x_low >= 0){
            v_lb = src + (y_high * src_width + x_low) * 3;
        }
        if (x_low < src_width){
            v_rb = src + (y_high * src_width + x_high) * 3;
        }
    }
    dst[index] = (w_lt * v_lt[2] + w_rt * v_rt[2] + w_lb * v_lb[2] + w_rb * v_rb[2]) / 255.f;
    dst[index + dst_num_pixels] = (w_lt * v_lt[1] + w_rt * v_rt[1] + w_lb * v_lb[1] + w_rb * v_rb[1]) / 255.f;
    dst[index + dst_num_pixels * 2] = (w_lt * v_lt[0] + w_rt * v_rt[0] + w_lb * v_lb[0] + w_rb * v_rb[0]) / 255.f;
}

void mat2tensor(float* src, float* dst, int channels, int src_height, int src_width, int dst_height, int dst_width, float padding_value){
    float scale = std::min(float(dst_height) / src_height, float(dst_width) / src_width);

    AffineMatrix2 src2dst, dst2src;

    src2dst.v_x[0] = scale;
    src2dst.v_x[1] = 0;
    src2dst.v_x[2] = - scale * src_width * 0.5 + dst_width * 0.5;
    src2dst.v_y[0] = 0;
    src2dst.v_y[1] = scale;
    src2dst.v_y[2] = -scale * src_height * 0.5 + dst_height * 0.5;

    dst2src.v_x[0] = 1 / scale;
    dst2src.v_x[1] = 0;
    dst2src.v_x[2] = - 1 / scale * dst_width * 0.5 + src_width * 0.5;
    dst2src.v_y[0] = 0;
    dst2src.v_y[1] = 1 / scale;
    dst2src.v_y[2] = -1/scale * dst_width * 0.5 + src_height * 0.5;

    dim3 blockSize(16, 16);
    dim3 gridSize((dst_width + blockSize.x -1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);
    mat2tensor_kernel<<<gridSize, blockSize>>>(src, dst, channels, src_height, src_width, dst_height, dst_width, padding_value, dst2src);
}

void mat2tensor_crop(float* src, float* dst, int channels, int src_height, int src_width, int dst_height, int dst_width, int crop_size, float padding_value) {
    // 计算缩放比例
    float scale = std::min(float(dst_height) / src_height, float(dst_width) / src_width);

    // 定义仿射变换矩阵
    AffineMatrix2 dst2src;

    // 设置仿射变换矩阵
    dst2src.v_x[0] = 1 / scale;  // x 方向的缩放
    dst2src.v_x[1] = 0;          // x 方向的旋转（无旋转）
    dst2src.v_x[2] = -1 / scale * crop_size / 2 + src_width / 2.0f;  // x 方向的偏移

    dst2src.v_y[0] = 0;          // y 方向的旋转（无旋转）
    dst2src.v_y[1] = 1 / scale;  // y 方向的缩放
    dst2src.v_y[2] = -1 / scale * crop_size / 2 + src_height / 2.0f; // y 方向的偏移



    // 设置 CUDA 核函数的线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((crop_size + blockSize.x - 1) / blockSize.x, (crop_size + blockSize.y - 1) / blockSize.y);

    // 调用 CUDA 核函数
    mat2tensor_kernel<<<gridSize, blockSize>>>(src, dst, channels, src_height, src_width, crop_size, crop_size, padding_value, dst2src);
}