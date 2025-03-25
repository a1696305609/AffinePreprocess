#!/bin/bash

# Fast compile cmake projects

DIR=cmake-build

if [ "$3" == "r" ]; then
    echo "remove $DIR"
    rm -fr $DIR
fi

if [ -d "$DIR" ]; then
    echo $DIR is a directory.
else
    mkdir $DIR
fi

cd $DIR

#UBTV=$(lsb_release -r)
#echo "ubuntu version: ${UBTV:0-5}"

#if [ "${UBTV:0-5}" == "24.04" ]; then
#    echo "copy 24.04/libvfvlog.so"
#else
#    cp ../lib/Ubuntu18.04/libvfvlog.so .
#fi

if [ "$2" == "d" ]; then
    echo "build Debug mode"
    cmake -DCMAKE_BUILD_TYPE=Debug ..
else
    echo "build Release mode"
    cmake -DCMAKE_BUILD_TYPE=Release ..
fi

cmake --build . -- -j$(nproc)

if [ "$?" == "0" ]; then
    echo "Build OK, execute..."
else
    echo "Build failed!"
    exit 1
fi

if [ "$1" == "r" ]; then
    echo "run ims_cv_demo"
    ./ims_cv_demo
fi
exit 0