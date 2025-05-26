cmake_minimum_required(VERSION 3.10)
project(sanity_checke LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG")
set(CMAKE_VERBOSE_MAKEFILE OFF)

# LibTorch
set(Torch_DIR "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/code/data_loader/lib/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(/usr/local/include)  # hoặc /usr/include nếu spng.h ở đó
link_directories(/usr/local/lib)


set(spng_DIR "/usr/local/lib/cmake/spng/" CACHE PATH "spng include directory")
find_package(SPNG REQUIRED)

# CUDA
find_package(CUDA REQUIRED)

# OpenCV
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4" CACHE PATH "OpenCV directory")
find_package(OpenCV REQUIRED)

# Threads and rt
find_package(Threads REQUIRED)
set(SPNG_LIBRARIES /usr/local/lib/libspng.so)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include
          /usr/local/include
          /opt/nvidia/tensorrt/include
          $ENV{TENSORRT_HOME}/include
)

find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS /usr/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib
          /opt/nvidia/tensorrt/lib
          $ENV{TENSORRT_HOME}/lib
)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS /usr/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib
          /opt/nvidia/tensorrt/lib
          $ENV{TENSORRT_HOME}/lib
)

find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin
    HINTS /usr/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib
          /opt/nvidia/tensorrt/lib
          $ENV{TENSORRT_HOME}/lib
)

set(TENSORRT_LIBRARIES 
    ${TENSORRT_LIBRARY_INFER} 
    ${TENSORRT_LIBRARY_ONNXPARSER} 
    ${TENSORRT_LIBRARY_PLUGIN}
)

if(NOT TENSORRT_LIBRARY_INFER OR NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT library not found. Please install TensorRT or set TENSORRT_HOME environment variable.")
endif()

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${SPNG_INCLUDE_DIR}
    /usr/local/cuda/include
)

set(LIBS
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${SPNG_LIBRARIES}
    Threads::Threads
    rt
)

add_executable(sanity_check sanity_check.cpp)
target_link_libraries(sanity_check PRIVATE ${LIBS})



message(STATUS "=== Build configuration ===")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries:   ${TORCH_LIBRARIES}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Threads libs:       ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "Realtime libs:      rt")
message(STATUS "RPATH:              ${Torch_DIR}/../lib")
message(STATUS "=== End of configuration ===")
