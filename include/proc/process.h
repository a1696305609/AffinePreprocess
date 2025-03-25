#pragma once

#ifndef NSB_PROC
#define NSB_PROC namespace proc{
#define NSE_PROC }
#define USE_PROC using namespace proc;
#endif

#include <opencv2/opencv.hpp>

NSB_PROC
class ims_proc
{
private:
    /* data */
public:
    ims_proc(/* args */){};
    ~ims_proc(){};
    void get_float_image_memory(const cv::Mat& image, float* dst, int channels, int height, int width);
    void preprocess(float* src, float* dst, int channels, int src_height, int src_width, int dst_height, int dst_width, int padding_value);
    void preprocess_expert(float* src, float* dst, int channels, int src_height, int src_width, int short_side, int crop_size, int padding_value);

    cv::Mat anti_preprocess(float* img_ptr, int channels, int height, int width);
};
NSE_PROC