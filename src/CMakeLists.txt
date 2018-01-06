# https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake
# https://cmake.org/cmake/help/v3.9/manual/cmake-buildsystem.7.html
# http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)
set(COMPILE_FLAGS "-Xcudafe \"--diag_suppress=set_but_not_used --diag_suppress=branch_past_initialization\"")
set(COMPILE_FLAGSO2 "-Xcicc -O0 -Xcudafe \"--diag_suppress=set_but_not_used --diag_suppress=branch_past_initialization\"")
set(arch 60)
include(CTest)
enable_testing()

include_directories(include)

# CORE
add_library(libcu.${arch} STATIC
  libcu/sentinel-msg.cpp
  libcu/sentinel.cpp
  libcu/sentinel-host.cpp
  libcu/cuda_runtime.cpp
  libcu/stringcu.cu
  libcu/libcu.cu
  libcu/libcu.stdlib.cu
  )
  set_source_files_properties(libcu/stringcu.cu libcu/libcu.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(libcu.${arch} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(libcu.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
if (M)
  set_target_properties(libcu.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
endif()


# TCL
add_library(libcu.tinytcl.${arch} STATIC
  libcu.tinytcl/libcu.tinytcl.cu

  #libcu.tinytcl/tclAlloc.cu
  #libcu.tinytcl/tclAssem.cu
  #libcu.tinytcl/tclBasic.cu
  #libcu.tinytcl/tclCmdAH.cu
  #libcu.tinytcl/tclCmdIL.cu
  #libcu.tinytcl/tclCmdMZ.cu
  #libcu.tinytcl/tclExpr.cu
  #libcu.tinytcl/tclGet.cu
  #libcu.tinytcl/tclGlob.cu
  #libcu.tinytcl/tclGpuAZ.cu
  #libcu.tinytcl/tclGpuEnv.cu
  #libcu.tinytcl/tclGpuStr.cu
  #libcu.tinytcl/tclGpuUtil.cu
  #libcu.tinytcl/tclHash.cu
  #libcu.tinytcl/tclHistory.cu
  #libcu.tinytcl/tclLoad-noop.cu
  #libcu.tinytcl/tclLoad.cu
  #libcu.tinytcl/tclObj.cu
  #libcu.tinytcl/tclParse.cu
  #libcu.tinytcl/tclProc.cu
  #libcu.tinytcl/tclUtil.cu
  #libcu.tinytcl/tclVar.cu
  #libcu.tinytcl/tclExDbg.cu
  #libcu.tinytcl/tclNoExtensions.cu
  #libcu.tinytcl/tclExGen.cu
  #libcu.tinytcl/tclMod.cu
  )
  set_source_files_properties(libcu.tinytcl/libcu.tinytcl.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGSO2}")
  target_compile_definitions(libcu.tinytcl.${arch} PRIVATE TCL_MEM_DEBUG)
  target_link_libraries(libcu.tinytcl.${arch} PRIVATE libcu.${arch})

include_directories(libcu.tinytcl)
add_executable(tinytcl.${arch}
	tinytcl/tinySh.cu
  )
  target_link_libraries(tinytcl.${arch} PRIVATE libcu.tinytcl.${arch})

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(libcu.tinytcl.${arch} PUBLIC cxx_std_11)
target_compile_features(tinytcl.${arch} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(libcu.tinytcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tinytcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
if (M)
  set_target_properties(libcu.tinytcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(tinytcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
endif()

