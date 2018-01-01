# https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake
# https://cmake.org/cmake/help/v3.9/manual/cmake-buildsystem.7.html
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)
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

add_library(libcu.tinytcl.${arch} STATIC
  #libcu.tinytcl/libcu.jimtcl.cu

  #libcu.tinytcl/tclAlloc.cu
  #libcu.tinytcl/tclAssem.cu
  #libcu.tinytcl/tclBasic.cu
  #libcu.tinytcl/tclCmdAH.cu
  #libcu.tinytcl/tclCmdIL.cu
  libcu.tinytcl/tclCmdMZ.cu
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
  target_link_libraries(libcu.tinytcl.${arch} PRIVATE libcu.${arch})

add_executable(tinytcl.${arch}
	tinytcl/tinySh.cu
  )
  target_link_libraries(tinytcl.${arch} PRIVATE libcu.jimtcl.${arch})

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(libcu.jimtcl.${arch} PUBLIC cxx_std_11)
target_compile_features(jimtcl.${arch} PUBLIC cxx_std_11)
target_compile_features(libcu.tinytcl.${arch} PUBLIC cxx_std_11)
target_compile_features(tinytcl.${arch} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(libcu.jimtcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(jimtcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(libcu.tinytcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tinytcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
if (M)
  set_target_properties(libcu.jimtcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(jimtcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(libcu.tinytcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(tinytcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
endif()
