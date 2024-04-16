set(PROJECT_NAME gpu_path_tracer)
set(SRCROOT ${PROJECT_SOURCE_DIR}/${PROJECT_NAME})
enable_language(CUDA)

set(SRC
    ${SRCROOT}/CudaPortable.cu
    # ${SRCROOT}/CudaPathTrace.h
    # ${SRCROOT}/CudaPathTrace.cu
    ${SRCROOT}/CudaRenderer.hpp
    # ${SRCROOT}/CudaRenderer.cpp
    ${SRCROOT}/CudaRenderer.cu
    ${SRCROOT}/main.cpp
    # add more here
)

add_project(${PROJECT_NAME}
            SOURCES ${SRC}
)

add_compile_definitions(
    GPU_PATH_TRACER
)

# setup cuda version
# SET(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT" )
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -dc")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -x cu -arch=sm_70 -dc")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -x cu -arch=sm_70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61")
# list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
