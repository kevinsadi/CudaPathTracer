cmake_minimum_required(VERSION 3.10)
project(TowardsRealtimeRayTracing)
set(CMAKE_CXX_STANDARD 17)

include(Macros.cmake)

# add external libraries
include_directories(${PROJECT_SOURCE_DIR}/external)

add_subdirectory(common)
add_subdirectory(cpu_path_tracer)

if (NOT APPLE)
    add_subdirectory(gpu_path_tracer)
endif()