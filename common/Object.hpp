//
// Created by LEI XU on 5/13/19.
//
#pragma once
#ifndef RAYTRACING_OBJECT_H
#define RAYTRACING_OBJECT_H

#include "Bounds3.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"

class Object
{
public:
    Object() {}
    virtual ~Object() {}
    FUNC_QUALIFIER inline virtual bool intersect(const Ray& ray) = 0;
    FUNC_QUALIFIER inline virtual bool intersect(const Ray& ray, float&, uint32_t&) const = 0;
    FUNC_QUALIFIER inline virtual Intersection getIntersection(Ray _ray) = 0;
    FUNC_QUALIFIER inline virtual void getSurfaceProperties(const glm::vec3&, const glm::vec3&, const uint32_t&, const glm::vec2&, glm::vec3&, glm::vec2&) const = 0;
    FUNC_QUALIFIER inline virtual glm::vec3 evalDiffuseColor(const glm::vec2&) const = 0;
    FUNC_QUALIFIER inline virtual Bounds3 getBounds() = 0;
    FUNC_QUALIFIER inline virtual float getArea() = 0;
    // Sample a point on the surface of the object, used for area light
    FUNC_QUALIFIER inline virtual void Sample(Intersection& pos, float& pdf) = 0;
    FUNC_QUALIFIER inline virtual bool hasEmit() = 0;
};



#endif //RAYTRACING_OBJECT_H
