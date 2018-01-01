# https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake
# https://cmake.org/cmake/help/v3.9/manual/cmake-buildsystem.7.html
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)
set(arch 60)
include(CTest)
enable_testing()

include_directories(include)

add_library(cu.${arch} STATIC
  libcu/sentinel-msg.cpp
  libcu/sentinel.cpp
  libcu/sentinel-host.cpp
  libcu/cuda_runtime.cpp
  libcu/stringcu.cu
  libcu/libcu.cu
  libcu/libcu.stdlib.cu
  )

add_library(cu.falloc.${arch} STATIC
  libcu.falloc/libcu.falloc.cu
  )

add_library(cu.fileutils.${arch} STATIC
  libcu.fileutils/libcu.fileutils.cu
  libcu.fileutils/sentinel-msg.cpp
  )
target_link_libraries(cu.fileutils.${arch} PRIVATE cu.${arch})

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(cu.${arch} PUBLIC cxx_std_11)
target_compile_features(cu.falloc.${arch} PUBLIC cxx_std_11)
target_compile_features(cu.fileutils.${arch} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
if (X64)
  set_target_properties(cu.${arch} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64" CUDA_64_BIT_DEVICE_CODE ON CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(cu.falloc.${arch} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64" CUDA_64_BIT_DEVICE_CODE ON CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(cu.fileutils.${arch} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64" CUDA_64_BIT_DEVICE_CODE ON CUDA_SEPARABLE_COMPILATION ON)
else()
  set_target_properties(cu.${arch} PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32" CUDA_64_BIT_DEVICE_CODE OFF CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(cu.falloc.${arch} PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32" CUDA_64_BIT_DEVICE_CODE OFF CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(cu.fileutils.${arch} PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32" CUDA_64_BIT_DEVICE_CODE OFF CUDA_SEPARABLE_COMPILATION ON)
endif()

if (BUILD_TESTING)
  add_executable(cu_tests
	libcu.tests/libcu.tests.cu
	libcu.tests/program.cu
  )

  if (X64)
    set_target_properties(cu_tests PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64" CUDA_64_BIT_DEVICE_CODE ON CUDA_SEPARABLE_COMPILATION ON)
  else()
    set_target_properties(cu_tests PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32" CUDA_64_BIT_DEVICE_CODE OFF CUDA_SEPARABLE_COMPILATION ON)
  endif()
  target_link_libraries(cu_tests PRIVATE cu.${arch} cu.falloc.${arch} cu.fileutils.${arch})

  add_test(NAME crtdefs_test1 COMMAND cu_tests 1)
  add_test(NAME ctype_test1 COMMAND cu_tests 2)
  add_test(NAME dirent_test1 COMMAND cu_tests 3)
  add_test(NAME errno_test1 COMMAND cu_tests 4)
  add_test(NAME falloc_lauched_cuda_kernel COMMAND cu_tests 5)
  add_test(NAME falloc_alloc_with_getchunk COMMAND cu_tests 6)
  add_test(NAME falloc_alloc_with_getchunks COMMAND cu_tests 7)
  add_test(NAME falloc_alloc_with_context COMMAND cu_tests 8)
  add_test(NAME fcntl_test1 COMMAND cu_tests 9)
  add_test(NAME grp_test1 COMMAND cu_tests 10)
  add_test(NAME pwd_test1 COMMAND cu_tests 11)
  add_test(NAME regex_test1 COMMAND cu_tests 12)
  add_test(NAME sentinel_test1 COMMAND cu_tests 13)
  add_test(NAME setjmp_test1 COMMAND cu_tests 14)
  add_test(NAME stdarg_parse COMMAND cu_tests 15)
  add_test(NAME stdarg_call COMMAND cu_tests 16)
  add_test(NAME stddef_test1 COMMAND cu_tests 17)
  add_test(NAME stdio_test1 COMMAND cu_tests 18)
  add_test(NAME stdio_64bit COMMAND cu_tests 19)
  add_test(NAME stdio_ganging COMMAND cu_tests 20)
  add_test(NAME stdio_scanf COMMAND cu_tests 21)
  add_test(NAME stdlib_test1 COMMAND cu_tests 22)
  add_test(NAME stdlib_strtol COMMAND cu_tests 23)
  add_test(NAME stdlib_strtoq COMMAND cu_tests 24)
  add_test(NAME string_test1 COMMAND cu_tests 25)
  add_test(NAME time_test1 COMMAND cu_tests 26)
  add_test(NAME unistd_test1 COMMAND cu_tests 27)

  if (APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath, so that the static cuda runtime can find it at runtime.
    set_property(TARGET cu_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()