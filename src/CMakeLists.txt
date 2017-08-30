cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)

add_library(libcu STATIC
  sentinel-msg.cpp
  sentinel.cpp
  sentinel-host.cpp
  host_functions.cpp
  libcu.cu
  libcu.stdlib.cu
  )

# Request that libcu be built with -std=c++11
# As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(libcu PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(libcu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
