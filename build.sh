#! /bin/sh
cp src/CMakeLists.work.txt src/CMakeLists.txt

rm -rf _build
mkdir _build
cd _build

mkdir build32
cd build32
cmake -DCMAKE_CUDA_FLAGS="-arch=sm_35" ../../src
cd ..

cmake --build build32 --config Debug

cd build32
ctest -C Debug

cd ../..
