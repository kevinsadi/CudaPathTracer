//
// Created by LEI XU on 5/13/19.
//
#pragma once
#ifndef RAYTRACING_VECTOR_H
#define RAYTRACING_VECTOR_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include "MathUtils.hpp"
#include "CudaPortable.hpp"

struct Vector3f {
public:
    float x, y, z;
    FUNC_QUALIFIER Vector3f() : x(0), y(0), z(0) {}
    FUNC_QUALIFIER Vector3f(float xx) : x(xx), y(xx), z(xx) {}
    FUNC_QUALIFIER Vector3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    FUNC_QUALIFIER Vector3f operator * (const float& r) const { return Vector3f(x * r, y * r, z * r); }
    FUNC_QUALIFIER Vector3f operator / (const float& r) const { return Vector3f(x / r, y / r, z / r); }

    FUNC_QUALIFIER float norm() { return glm::sqrt(x * x + y * y + z * z); }
    FUNC_QUALIFIER Vector3f normalized() {
        float n = glm::sqrt(x * x + y * y + z * z);
        return Vector3f(x / n, y / n, z / n);
    }

    FUNC_QUALIFIER Vector3f operator * (const Vector3f& v) const { return Vector3f(x * v.x, y * v.y, z * v.z); }
    FUNC_QUALIFIER Vector3f operator - (const Vector3f& v) const { return Vector3f(x - v.x, y - v.y, z - v.z); }
    FUNC_QUALIFIER Vector3f operator + (const Vector3f& v) const { return Vector3f(x + v.x, y + v.y, z + v.z); }
    FUNC_QUALIFIER Vector3f operator - () const { return Vector3f(-x, -y, -z); }
    FUNC_QUALIFIER Vector3f& operator += (const Vector3f& v) { x += v.x, y += v.y, z += v.z; return *this; }
    FUNC_QUALIFIER Vector3f& operator *= (const Vector3f& v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
    FUNC_QUALIFIER friend Vector3f operator * (const float& r, const Vector3f& v)
    {
        return Vector3f(v.x * r, v.y * r, v.z * r);
    }
    FUNC_QUALIFIER friend std::ostream& operator << (std::ostream& os, const Vector3f& v)
    {
        return os << v.x << ", " << v.y << ", " << v.z;
    }
    FUNC_QUALIFIER double       operator[](int index) const;
    FUNC_QUALIFIER double& operator[](int index);


    FUNC_QUALIFIER static Vector3f Min(const Vector3f& p1, const Vector3f& p2) {
        return Vector3f(glm::min(p1.x, p2.x), glm::min(p1.y, p2.y),
            glm::min(p1.z, p2.z));
    }

    FUNC_QUALIFIER static Vector3f Max(const Vector3f& p1, const Vector3f& p2) {
        return Vector3f(glm::max(p1.x, p2.x), glm::max(p1.y, p2.y),
            glm::max(p1.z, p2.z));
    }

    FUNC_QUALIFIER inline glm::vec3 toGlm() const {
        return glm::vec3(x, y, z);
    }

    CUDA_PORTABLE(Vector3f);
};
FUNC_QUALIFIER inline double Vector3f::operator[](int index) const {
    return (&x)[index];
}


struct Vector2f
{
public:
    FUNC_QUALIFIER Vector2f() : x(0), y(0) {}
    FUNC_QUALIFIER Vector2f(float xx) : x(xx), y(xx) {}
    FUNC_QUALIFIER Vector2f(float xx, float yy) : x(xx), y(yy) {}
    FUNC_QUALIFIER Vector2f operator * (const float& r) const { return Vector2f(x * r, y * r); }
    FUNC_QUALIFIER Vector2f operator + (const Vector2f& v) const { return Vector2f(x + v.x, y + v.y); }
    float x, y;

    CUDA_PORTABLE(Vector2f);
};

FUNC_QUALIFIER inline Vector3f lerp(const Vector3f& a, const Vector3f& b, const float& t)
{
    return a * (1 - t) + b * t;
}

FUNC_QUALIFIER inline Vector3f normalize(const Vector3f& v)
{
    float mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (mag2 > 0) {
        float invMag = 1 / sqrtf(mag2);
        return Vector3f(v.x * invMag, v.y * invMag, v.z * invMag);
    }

    return v;
}

FUNC_QUALIFIER inline float dotProduct(const Vector3f& a, const Vector3f& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

FUNC_QUALIFIER inline Vector3f crossProduct(const Vector3f& a, const Vector3f& b)
{
    return Vector3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

FUNC_QUALIFIER inline Vector3f fromGlm (const glm::vec3& v)
{
    return Vector3f(v.x, v.y, v.z);
}


#endif //RAYTRACING_VECTOR_H
