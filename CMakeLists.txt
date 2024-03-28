cmake_minimum_required(VERSION 3.10)
project(TowardsRealtimeRayTracing)
set(CMAKE_CXX_STANDARD 17)

include(Macros.cmake)

add_subdirectory(common)
add_subdirectory(cpu_path_tracer)
add_subdirectory(gpu_path_tracer)