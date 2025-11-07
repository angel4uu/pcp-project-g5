cmake_minimum_required(VERSION 3.15)
project(YOLO_TensorRT_CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Opciones de compilación
option(ENABLE_CUDA "Usar CUDA" ON)
option(ENABLE_TENSORRT "Usar TensorRT" ON)

# Encontrar paquetes
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT (ajusta la ruta según tu instalación)
set(TENSORRT_ROOT "/usr/local/TensorRT" CACHE PATH "Ruta a TensorRT")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")

# Configuración de includes
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# Configuración de librerías
link_directories(
    ${TENSORRT_LIB_DIR}
)

# Target: YOLO TensorRT Detector
add_executable(yolo_tensorrt_detector
    yolo_tensorrt_detector.cpp
)

target_link_libraries(yolo_tensorrt_detector
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvparsers
    nvcaffe_parser
)

# Flags de compilación para CUDA
set_target_properties(yolo_tensorrt_detector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(ENABLE_CUDA)
    set_target_properties(yolo_tensorrt_detector PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Opciones de compilación de optimización
target_compile_options(yolo_tensorrt_detector PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast>
)

# Mensaje de configuración
message(STATUS "================================")
message(STATUS "YOLO TensorRT CUDA Detector")
message(STATUS "================================")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_ROOT}")
message(STATUS "================================")

# Instrucciones de compilación
message(STATUS "")
message(STATUS "Para compilar:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DTENSORRT_ROOT=/ruta/a/TensorRT")
message(STATUS "  make -j\$(nproc)")
message(STATUS "")
