# AffinePreprocess
A C++ TensorRt preprocess of yolov5 and middle crop
---
This is a test project when I am deploying Yolov5  use c++, cuda, TensorRT

## My Enviroment
Ubuntu24.04

gcc-9 g++-9

cmake-3.28

cuda-11.3

cudnn-8.6.0

TensorRT-8.5.3.1

opencv-4.6.0

## Use

check src/demo/main.cpp for logitis

change engine path and image path

At first, you must build project

```sh
cd AffinePreprecess
./build.sh
```

Then run demo

```sh
./cmake-build/proc_demo
```
