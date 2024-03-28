cmake_minimum_required(VERSION 3.10)
project(RayTracing)

set(CMAKE_CXX_STANDARD 17)

# Add the compile flags
set(CMAKE_CXX_FLAGS "-O3")

if (APPLE)
    set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
    set(OPENMP_LIBRARIES "/usr/local/Cellar/libomp/17.0.6/lib")
    set(OPENMP_INCLUDES "/usr/local/Cellar/libomp/17.0.6/include")
else() # windows

endif()

OPTION (USE_OpenMP "Use OpenMP to enable <omp.h>" ON)

# Find OpenMP
if(USE_OpenMP)
    if (APPLE)
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-fopenmp -Wno-unused-command-line-argument")
            set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
            set(OpenMP_CXX_FLAGS "-fopenmp -Wno-unused-command-line-argument")
            set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        endif()
    else() # windows

    endif()
endif()

if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
endif(USE_OpenMP)

if (OPENMP_FOUND)
    if (APPLE)
        include_directories("${OPENMP_INCLUDES}")
        link_directories("${OPENMP_LIBRARIES}")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else() # windows

    endif()
endif(OPENMP_FOUND)


add_executable(RayTracing main.cpp Object.hpp Vector.cpp Vector.hpp Sphere.hpp global.hpp Triangle.hpp Scene.cpp
        Scene.hpp Light.hpp AreaLight.hpp BVH.cpp BVH.hpp Bounds3.hpp Ray.hpp Material.hpp Intersection.hpp
        Renderer.cpp Renderer.hpp)