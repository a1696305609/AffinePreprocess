
cmake_minimum_required(VERSION 3.10) # 应该修改为3.12, 由于Clang 的支持问题

set(CMAKE_C_COMPILER "/usr/bin/gcc") # 设置C编译器
set(CMAKE_CXX_COMPILER "/usr/bin/g++") # 设置C++编译器


project(PROC VERSION 0.2.0) # 设置自定义软件包名

# config  CUDA
#=========================================
#寻找cuda
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) # 设置CUDA编程编译器
enable_language(CUDA) # 使项目可用CUDA编程语言

message(STATUS "CUDA ${CUDA_VERSION}")
message(STATUS "CUDA ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA${CUDA_INCLUDE_DIRS}")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
# no lib dirs  varienty

# 设置 CUDA 的包含目录和库目录
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
# set(CUDA_LIB_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

# CUDA_FOUND：表示是否成功找到了CUDA工具包。
# CUDA_VERSION_MAJOR：CUDA的主要版本号。
# CUDA_VERSION_MINOR：CUDA的次要版本号。
# CUDA_VERSION或CUDA_VERSION_STRING：CUDA的完整版本号，格式为X.Y。
# CUDA_TOOLKIT_ROOT_DIR：CUDA工具包的安装路径。
# CUDA_INCLUDE_DIRS：包含CUDA头文件的目录。
# CUDA_LIBRARIES：CUDA运行时库。
# CUDA_CUFFT_LIBRARIES：用于CUDA FFT实现的库。
# CUDA_CUBLAS_LIBRARIES：用于CUDA BLAS实现的库.
# CUDA_cudart_static_LIBRARY：静态可链接的CUDA运行时库（仅适用于CUDA版本5.5及以上）.
# CUDA_cudadevrt_LIBRARY：设备运行时库，可分离编译需要（从3.7版本开始）.
# CUDA_cupti_LIBRARY：CUDA工具包的性能分析工具接口库

include_directories(CUDA_INCLUDE_DIRS)

# Config TensorRT
# =============================================================
# find_package(TensorRT REQUIRED)
# if (TensorRT_FOUND)
#     message(STATUS "TensorRT ${TensorRT_VERSION}")
#     message(STATUS "TensorRT ${TensorRT_INCLUDE_DIRS}")
#     message(STATUS "TensorRT ${TensorRT_LIBRARIES}")
    # include_directories(TensorRT_INCLUDE_DIRS)
    # link_directories(TensorRT_LIBRARIES)

    
# TensorRT_FOUND：表示是否成功找到了TensorRT库
# TensorRT_VERSION：TensorRT的版本号
# TensorRT_INCLUDE_DIRS：包含TensorRT头文件的目录
# TensorRT_LIBRARIES：TensorRT的库文件
# TensorRT_LIBRARY_DIRS：TensorRT库文件所在的目录

# can't find, so add static path
include_directories(/usr/local/TensorRT-8.5.3.1/include/)
link_directories(/usr/local/TensorRT-8.5.3.1/lib/)

# config c++ flags
#===================================================================
add_library(cxx_flags INTERFACE)
target_compile_features(cxx_flags INTERFACE cxx_std_17)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
target_compile_options(cxx_flags INTERFACE
                       "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
                       "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
                      )


# config c++ definitions
# ===========================================================================
string(TIMESTAMP COMPILE_TIME %Y-%m-%d_%H:%M:%S)
set(build_time ${COMPILE_TIME})
configure_file(CMakeConfig.h.in CMakeConfig.h)


# find packages
# ========================================================================
find_package(Threads REQUIRED)

# ========================================================================
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
# set(OpenCV_DIR "~/opencv/build/installed/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION}")
message(STATUS "OpenCV ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV ${OpenCV_LIBS}")
#opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio

message(STATUS "Begin building...")

# yolos set up
# =================================================================
include_directories(${PROJECT_SOURCE_DIR}/src/yolo/)
file(GLOB_RECURSE YOLO_SRCS ${PROJECT_SOURCE_DIR}/src/yolo/*.cpp ${PROJECT_SOURCE_DIR}/src/yolo/*.cu)

#  myPlugin set up
# =================================================================
#include_directories(${PROJECT_SOURCE_DIR}/plugin/)
#file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

# add_library(myplugins SHARED ${PLUGIN_SRCS})
# target_link_libraries(myplugins nvinfer cudart)

# libs
#=======================================================================
# lib_ims
add_library(proc
    src/proc/process.cpp
    ${YOLO_SRCS}
     )
target_include_directories(proc PUBLIC
                            "${CUDA_INCLUDE_DIRS}"  # 添加的CUDA库
                            "${OpenCV_INCLUDE_DIRS}"
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/include")
# executables
#=======================================================================
# ims_cv_demo
add_executable(proc_demo src/demo/main.cpp)

target_link_libraries(proc_demo PUBLIC cxx_flags
    ${OpenCV_LIBS}
    #opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio
    # ${CUDA_LIBRARIES}  # 添加的CUDA库
    proc
    Threads::Threads
    ${CUDA_cudadevrt_LIBRARY} # cudart
    nvinfer # /usr/local/TensorRT-8.2.1.8/targets/x86_64-linux-gnu/lib/libnvinfer.so
)

# target_include_directories(ims_cv_demo PUBLIC
#                              "${OpenCV_INCLUDE_DIRS}"
#                             #  "${CUDA_INCLUDE_DIRS}"
#                              "${PROJECT_SOURCE_DIR}/include")

# # test_parallel
# add_executable(test_parallel src/ims_cv_demo/test_parallel.cpp)

# target_link_libraries(test_parallel PUBLIC cxx_flags
#    ${OpenCV_LIBS}
#     # ${CUDA_LIBRARIES}  # 添加的CUDA库
#     ims_cv
#     Threads::Threads
#     cudart
# )

# target_include_directories(test_parallel PUBLIC
#                              "${OpenCV_INCLUDE_DIRS}"
#                             #  "${CUDA_INCLUDE_DIRS}"
#                              "${PROJECT_SOURCE_DIR}/include")
