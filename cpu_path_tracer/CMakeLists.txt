set(PROJECT_NAME cpu_path_tracer)
set(SRCROOT ${PROJECT_SOURCE_DIR}/${PROJECT_NAME})

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Set the compiler to clang++ from LLVM
    set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")

    # Add the compile flag -fopenmp
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    find_package(OpenMP REQUIRED)
    # Add the compile flag -fopenmp
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
else()
    find_package(OpenMP REQUIRED)
    # on windows we use cl.exe
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /O2")
endif()

set(SRC
    ${SRCROOT}/main.cpp
    # add more here
)

add_project(${PROJECT_NAME}
            SOURCES ${SRC}
)

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
