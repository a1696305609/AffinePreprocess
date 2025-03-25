#include <iostream>
#include <proc/process.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h> // 添加了cuda库分配内存

#include "preprocess.h"
#include "cuda_utils.h"
#include <cmath>

USE_PROC

void ims_proc::get_float_image_memory(const cv::Mat &image, float* dst, int channels, int height, int width){
    uint8_t* src_cpu = image.data;
    int len_data = channels * height * width;
    uint8_t* src;
    CUDA_CHECK(cudaMalloc(&src, len_data * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(src, src_cpu, len_data * sizeof(uint8_t), cudaMemcpyHostToDevice));
    uint8_2_float(src, dst, channels, height, width);
    CUDA_CHECK(cudaFree(src));
}

void ims_proc::preprocess(float* src, float* dst, int channels, int src_height, int src_width, int dst_height, int dst_width, int padding_value){
    mat2tensor(src, dst, channels, src_height, src_width, dst_height, dst_width, float(padding_value));
    return;
}

void ims_proc::preprocess_expert(float* src, float* dst, int channels, int src_height, int src_width, int short_side, int crop_size, int padding_value){
    float scale = std::max(float(short_side) / src_height, float(short_side) / src_width);
    
    int dst_height = int(src_height * scale);
    int dst_width = int(src_width * scale);

    mat2tensor_crop(src, dst, channels, src_height, src_width, dst_height, dst_width, crop_size, (float)padding_value);

    return;
}

cv::Mat ims_proc::anti_preprocess(float *img_ptr, int channels, int height, int width){
    cv::Mat blob(1, channels * height * width, CV_32F, img_ptr); // 1D Mat
    blob = blob.reshape(1, {channels, height, width}); // 转换为 3D Mat
    cv::Mat image(height, width, CV_32FC(channels));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // 获取 Blob 中的像素值
                float pixelValue = blob.at<float>(c, h, w);

                // 将像素值存储到图像中
                image.at<cv::Vec3f>(h, w)[c] = pixelValue;
            }
        }
    }

    // 将像素值从 [0, 1] 恢复到 [0, 255]
    image.convertTo(image, CV_8UC3, 255.0);
    cv::Mat img;
    cvtColor(image, img, cv::COLOR_RGB2BGR);

    return img;
}
