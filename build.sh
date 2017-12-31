#! /bin/sh
cp src/CMakeLists.work.txt src/CMakeLists.txt

rm -rf _build
mkdir _build
cd _build

# BUILD32
mkdir build32
cd build32
cmake -DX64=0 -DCMAKE_CUDA_FLAGS="-arch=sm_60" ../../src
cd ..

cmake --build build32 --config Debug

cd build32
#ctest -C Debug
cd ..

# BUILD64
mkdir build64
cd build64
cmake -DX64=1 -DCMAKE_CUDA_FLAGS="-arch=sm_60" ../../src
cd ..

cmake --build build64 --config Debug

cd build64
#ctest -C Debug
cd ..

cd ..
