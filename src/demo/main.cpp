#include <cstdio>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
#include <proc/process.h>
#include <cuda_utils.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

using namespace std;
using namespace proc;
using namespace cv;
using namespace nvinfer1;


int main(){
    CUDA_CHECK(cudaSetDevice(0));

    ims_proc processed;

    string file_name = "/home/xp/projects/test_preprocess/test_image/01-46-1740441706069-pre.jpg";

    Mat image = imread(file_name, IMREAD_COLOR);

    image.convertTo(image, CV_32F, 1.f);
    float* look = image.ptr<float>();


    float* input_buffer;
    float* output_buffer;
    float* cpu_output_buffer;
    cudaMalloc(&input_buffer, image.channels() * image.rows * image.cols * sizeof(float));
    cudaMalloc(&output_buffer, 224 * 224 * 3 * sizeof(float));
    cudaMallocHost(&cpu_output_buffer, 224 * 224 * 3 * sizeof(float));

    cudaMemcpy(input_buffer, image.data, image.channels() * image.rows * image.cols * sizeof(float), cudaMemcpyHostToDevice);

    processed.preprocess_expert(input_buffer, output_buffer, image.channels(), image.rows, image.cols, 256, 224, 0);

    cudaMemcpy(cpu_output_buffer, output_buffer, 224 * 224 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    Mat recovery_img = processed.anti_preprocess(cpu_output_buffer, 3, 224, 224);

    imwrite("test_image/crop_size.jpg", recovery_img);
    
    return 0;
}