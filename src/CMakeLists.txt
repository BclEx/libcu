# https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake
# https://cmake.org/cmake/help/v3.9/manual/cmake-buildsystem.7.html
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)
set(COMPILE_FLAGS "-Xcudafe \"--diag_suppress=branch_past_initialization\"")
set(arch 60)
include(CTest)
enable_testing()

include_directories(include)

# TCL
add_library(libcu.jimtcl.${arch} STATIC
  #libcu.jimtcl/libcu.jimtcl.cu

  libcu.jimtcl/jim-interactive.cu
  libcu.jimtcl/init.cu
  libcu.jimtcl/jim.cu
  libcu.jimtcl/jim-aio.cu
  libcu.jimtcl/jim-array.cu
  libcu.jimtcl/jim-clock.cu
  libcu.jimtcl/jim-eventloop.cu
  libcu.jimtcl/jim-exec.cu
  libcu.jimtcl/jim-file.cu
  libcu.jimtcl/jim-format.cu
  libcu.jimtcl/jim-gpu.cu
  libcu.jimtcl/jim-history.cu
  libcu.jimtcl/jim-load.cu
  libcu.jimtcl/jim-namespace.cu
  libcu.jimtcl/jim-pack.cu
  libcu.jimtcl/jim-package.cu
  libcu.jimtcl/jim-readdir.cu
  libcu.jimtcl/jim-regexp.cu
  libcu.jimtcl/jim-subcmd.cu
  #_libcu.jimtcl/jim-tclprefix.cu
  libcu.jimtcl/jimex.cu
  libcu.jimtcl/utf8.cu
  )
  target_link_libraries(libcu.jimtcl.${arch} PRIVATE libcu.${arch})

add_executable(jimtcl.${arch}
	jimtcl/jimsh.cu
  )
  target_link_libraries(jimtcl.${arch} PRIVATE libcu.jimtcl.${arch})

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(libcu.jimtcl.${arch} PUBLIC cxx_std_11)
target_compile_features(jimtcl.${arch} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(libcu.jimtcl.${arch} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}" PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(jimtcl.${arch} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}" PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
if (M)
  set_target_properties(libcu.jimtcl.${arch} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(jimtcl.${arch} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
endif()
