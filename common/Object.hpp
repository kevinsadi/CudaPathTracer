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
    float area;
    Object() {}
    virtual ~Object() {}
    // ---------Edtior Only----------
    // you are free to use virutal method for convenience
    FUNC_QUALIFIER inline virtual Bounds3 getBounds() = 0;

    // ---------Runtime---------
    // for runtime method, you can't directly use virtual function, you need explicity 
    // convert Object* to Triangle* or MeshTriangle* and call the method
    FUNC_QUALIFIER inline virtual bool intersect(const Ray& ray, float&, uint32_t&) const = 0;
};



#endif //RAYTRACING_OBJECT_H
