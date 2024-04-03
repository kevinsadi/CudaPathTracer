//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_INTERSECTION_H
#define RAYTRACING_INTERSECTION_H
#include "Material.hpp"
#include "CudaPortable.hpp"
#include "MathUtils.hpp"

class Object;
// class Sphere;

struct Intersection
{
    FUNC_QUALIFIER inline Intersection()
    {
        happened = false;
        coords = glm::vec3();
        normal = glm::vec3();
        distance = kDoubleInfinity;
        // obj = nullptr;
    }
    bool happened;
    glm::vec3 coords;
    glm::vec3 tcoords;
    glm::vec3 normal;
    glm::vec3 emit;
    double distance;
    float triangleArea;
    Material m;
};
#endif // RAYTRACING_INTERSECTION_H
