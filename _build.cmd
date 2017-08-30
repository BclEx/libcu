@echo off
rm -rf _build
mkdir _build
cd _build

mkdir build32 & pushd build32
cmake -G "Visual Studio 12 2013" -DCMAKE_CUDA_FLAGS="-arch=sm_35" ../../src
popd
mkdir build64 & pushd build64
cmake -G "Visual Studio 12 2013 Win64" -DCMAKE_CUDA_FLAGS="-arch=sm_35" ../../src
popd
cmake --build build32 --config Release
cmake --build build64 --config Release


