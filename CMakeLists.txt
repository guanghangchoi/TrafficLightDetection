cmake_minimum_required(VERSION 3.0.2)

add_compile_options(-std=c++14)

project(traffic_light_test)

set(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(CUDA)
if (CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib)
  message(STATUS "CUDA is available!")
  message(STATUS "CUDA Libs: ${CUDA_LIBRARIES}")
  message(STATUS "CUDA Headers: ${CUDA_INCLUDE_DIRS}")
else ()
  message(STATUS "CUDA NOT FOUND")
endif ()

find_library(NVINFER NAMES nvinfer)
find_library(NVONNXPARSER nvonnxparser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
if(NVINFER AND NVONNXPARSER AND NVINFER_PLUGIN)
  message(STATUS "TensorRT is available!")
  message(STATUS "NVINFER: ${NVINFER}")
  message(STATUS "NVPARSERS: ${NVPARSERS}")
  message(STATUS "NVINFER_PLUGIN: ${NVINFER_PLUGIN}")
  message(STATUS "NVONNXPARSER: ${NVONNXPARSER}")
else()
  message(STATUS "TensorRT is NOT Available")
endif()

find_library(CUDNN_LIBRARY
NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
PATH_SUFFIXES lib lib64 bin
DOC "CUDNN library." )
if(CUDNN_LIBRARY)
  message(STATUS "CUDNN is available!")
  message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
else()
  message(STATUS "CUDNN is NOT Available")
endif()

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${Opencv_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-O2 -Wall -Wunused-variable ${CMAKE_CXX_FLAGS} -fpic -std=c++11 -pthread")
add_executable(main
  src/main.cpp
)

target_link_libraries(main
  ${OpenCV_LIBS}
)

add_library(trt 
  lib/src/trt.cpp)

target_include_directories(trt PUBLIC
  lib/include
)
target_link_libraries(trt
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARY}
)
target_link_libraries(main
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARY}
    trt
)