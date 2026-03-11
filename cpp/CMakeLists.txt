cmake_minimum_required(VERSION 3.20)
project(yolo_pipeline_cpp LANGUAGES CXX)

option(PIPELINE_ENABLE_ORT "Enable ONNX Runtime backend" ON)
option(PIPELINE_ENABLE_TENSORRT "Enable TensorRT backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(yaml-cpp REQUIRED)

add_executable(pipeline_benchmark
    src/main.cpp
)

target_include_directories(pipeline_benchmark PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pipeline_benchmark PRIVATE ${OpenCV_LIBS} yaml-cpp)

if(PIPELINE_ENABLE_ORT)
    if(NOT DEFINED ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "PIPELINE_ENABLE_ORT=ON requires -DONNXRUNTIME_ROOT=<path>")
    endif()

    target_compile_definitions(pipeline_benchmark PRIVATE PIPELINE_ENABLE_ORT=1)
    target_include_directories(pipeline_benchmark PRIVATE ${ONNXRUNTIME_ROOT}/include)
    target_link_directories(pipeline_benchmark PRIVATE ${ONNXRUNTIME_ROOT}/lib)
    target_link_libraries(pipeline_benchmark PRIVATE onnxruntime)
endif()

if(PIPELINE_ENABLE_TENSORRT)
    if(NOT DEFINED TENSORRT_ROOT)
        message(FATAL_ERROR "PIPELINE_ENABLE_TENSORRT=ON requires -DTENSORRT_ROOT=<path>")
    endif()

    target_compile_definitions(pipeline_benchmark PRIVATE PIPELINE_ENABLE_TENSORRT=1)
    target_include_directories(pipeline_benchmark PRIVATE ${TENSORRT_ROOT}/include)
    target_link_directories(pipeline_benchmark PRIVATE ${TENSORRT_ROOT}/lib)
    target_link_libraries(pipeline_benchmark PRIVATE nvinfer nvonnxparser cudart)
endif()
