@echo off
copy /Y src\CMakeLists.full.txt src\CMakeLists.txt

rem START
rd /s /q _build
rem rm -rf _build
mkdir _build
pushd _build


rem BUILD64
mkdir build64
pushd build64
cmake -G "Visual Studio 11 2012 Win64" -DCMAKE_CUDA_FLAGS="-arch=sm_60" ../../src
popd
cmake --build build64 --config Debug
pushd build64
ctest -C Debug
popd


rem BUILD32
mkdir build32
pushd build32
rem cmake -G "Visual Studio 11 2012" -DCMAKE_CUDA_FLAGS="-arch=sm_60" ../../src
popd
rem cmake --build build32 --config Debug
pushd build32
rem ctest -C Debug
popd


rem END
popd