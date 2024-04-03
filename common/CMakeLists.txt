# clean cache at every build
set(COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/common/CudaPortable.hpp
    ${PROJECT_SOURCE_DIR}/common/OBJ_Loader.hpp
    ${PROJECT_SOURCE_DIR}/common/Object.hpp
    ${PROJECT_SOURCE_DIR}/common/Sphere.hpp
    ${PROJECT_SOURCE_DIR}/common/Triangle.hpp
    ${PROJECT_SOURCE_DIR}/common/Triangle.cpp
    ${PROJECT_SOURCE_DIR}/common/Scene.cpp
    ${PROJECT_SOURCE_DIR}/common/Scene.hpp
    ${PROJECT_SOURCE_DIR}/common/BVH.cpp
    ${PROJECT_SOURCE_DIR}/common/BVH.hpp
    ${PROJECT_SOURCE_DIR}/common/Bounds3.hpp
    ${PROJECT_SOURCE_DIR}/common/Ray.hpp
    ${PROJECT_SOURCE_DIR}/common/Material.hpp
    ${PROJECT_SOURCE_DIR}/common/Intersection.hpp
    ${PROJECT_SOURCE_DIR}/common/Renderer.cpp
    ${PROJECT_SOURCE_DIR}/common/Renderer.hpp
    ${PROJECT_SOURCE_DIR}/common/Utility.cpp
    ${PROJECT_SOURCE_DIR}/common/Utility.hpp
    ${PROJECT_SOURCE_DIR}/common/MathUtils.hpp
    ${PROJECT_SOURCE_DIR}/common/MathUtils.cpp
    CACHE INTERNAL ""
)