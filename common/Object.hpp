//
// Created by LEI XU on 5/13/19.
//
#pragma once
#ifndef RAYTRACING_OBJECT_H
#define RAYTRACING_OBJECT_H

#include "Vector.hpp"
#include "Bounds3.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"

class Object
{
public:
    Object() {}
    virtual ~Object() {}
    FUNC_QUALIFIER virtual bool intersect(const Ray& ray) = 0;
    FUNC_QUALIFIER virtual bool intersect(const Ray& ray, float&, uint32_t&) const = 0;
    FUNC_QUALIFIER virtual Intersection getIntersection(Ray _ray) = 0;
    FUNC_QUALIFIER virtual void getSurfaceProperties(const Vector3f&, const Vector3f&, const uint32_t&, const Vector2f&, Vector3f&, Vector2f&) const = 0;
    FUNC_QUALIFIER virtual Vector3f evalDiffuseColor(const Vector2f&) const = 0;
    FUNC_QUALIFIER virtual Bounds3 getBounds() = 0;
    FUNC_QUALIFIER virtual float getArea() = 0;
    // Sample a point on the surface of the object, used for area light
    FUNC_QUALIFIER virtual void Sample(Intersection& pos, float& pdf) = 0;
    FUNC_QUALIFIER virtual bool hasEmit() = 0;
};



#endif //RAYTRACING_OBJECT_H
