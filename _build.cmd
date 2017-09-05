@echo off
rd /s /q _build
rem rm -rf _build
mkdir _build
pushd _build

mkdir build32 & pushd build32
cmake -G "Visual Studio 11 2012" -DCMAKE_CUDA_FLAGS="-arch=sm_35" ../../src
popd
rem mkdir build64 & pushd build64
rem cmake -G "Visual Studio 11 2012 Win64" -DCMAKE_CUDA_FLAGS="-arch=sm_35" ../../src
rem popd
cmake --build build32 --config Debug
rem cmake --build build64 --config Debug

pushd build32
ctest -C Debug
popd

popd