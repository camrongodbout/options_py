cmake_minimum_required(VERSION 3.22)
project(cuda_options LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

include(FetchContent)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.6.1  # Optionally, specify a version tag
)
FetchContent_MakeAvailable(pybind11)

file(GLOB_RECURSE LIB_SOURCES
        "src/*.cpp"
        "src/**/*.cpp"
        "src/*.cu"
        "src/**/*.cu"
)

include_directories(src)
include_directories(src/nodes)

add_library(cuda_options SHARED ${LIB_SOURCES})
target_link_libraries(cuda_options PRIVATE ${CUDAToolkit_LIBRARIES})

pybind11_add_module(options_py bindings/bindings.cpp)
target_link_libraries(options_py PRIVATE cuda_options pybind11::module ${CUDAToolkit_LIBRARIES})

set_target_properties(cuda_options PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80"  # Change to your GPU's compute capability
)