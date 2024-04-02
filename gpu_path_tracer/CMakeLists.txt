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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70")