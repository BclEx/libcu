@echo off
rmdir /S/Q _build
mkdir _build
cd _build
rm -rf *
cmake -DCMAKE_CUDA_FLAGS="-arch=sm_35" ../src