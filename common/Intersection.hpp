//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_INTERSECTION_H
#define RAYTRACING_INTERSECTION_H
#include "Vector.hpp"
#include "Material.hpp"
#include "CudaPortable.hpp"
#include "MathUtils.hpp"
class Object;
class Sphere;

struct Intersection
{
    FUNC_QUALIFIER Intersection() {
        happened = false;
        coords = Vector3f();
        normal = Vector3f();
        distance = kDoubleInfinity;
        obj = nullptr;
        m = nullptr;
    }
    bool happened;
    Vector3f coords;
    Vector3f tcoords;
    Vector3f normal;
    Vector3f emit;
    double distance;
    Object* obj;
    Material* m;
};
#endif //RAYTRACING_INTERSECTION_H
