#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "CudaPortable.hpp"

#define Pi 3.1415926535897932384626422832795028841971f
#define PiTwo 6.2831853071795864769252867665590057683943f
#define PiInv 1.f / Pi
#define Epsilon 1e-5f

#define kDoubleInfinity 1e30f
#define kDoubleNegInfinity -kDoubleInfinity
#define kFloatInfinity 3.402823466e+38F
#define kFloatNegInfinity -kFloatInfinity

FUNC_QUALIFIER inline void swap(float &a, float &b)
{
    float temp = a;
    a = b;
    b = temp;
}

FUNC_QUALIFIER inline float clamp(const float &lo, const float &hi, const float &v)
{
    return glm::max(lo, glm::min(hi, v));
}

FUNC_QUALIFIER inline bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0)
        return false;
    else if (discr == 0)
        x0 = x1 = -0.5 * b / a;
    else
    {
        float q = (b > 0) ? -0.5 * (b + sqrt(discr)) : -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1)
        swap(x0, x1);
    return true;
}

FUNC_QUALIFIER inline float get_random_float()
{
#ifdef GPU_PATH_TRACER
    return 0.5f; // todo
#else
    // return 0.5f;// todo
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_real_distribution<float> dist(0.f, 1.f); // distribution in range [1, 6]

    return dist(rng);
#endif
}

static std::string vec3ToString(const glm::vec3 &vec)
{
    std::stringstream ss;
    ss << "{ x = " << vec.x << ", y = " << vec.y << ", z = " << vec.z << " }";
    return ss.str();
}

namespace Math
{
    FUNC_QUALIFIER glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale){
        glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
        glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * Pi / 180.f, glm::vec3(1.f, 0.f, 0.f));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * Pi / 180.f, glm::vec3(0.f, 1.f, 0.f));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * Pi / 180.f, glm::vec3(0.f, 0.f, 1.f));
        glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
        return translationMat * rotationMat * scaleMat;
    }
    template <typename T>
    FUNC_QUALIFIER bool between(const T &x, const T &min, const T &max)
    {
        return x >= min && x <= max;
    }

    FUNC_QUALIFIER inline float satDot(glm::vec3 a, glm::vec3 b)
    {
        return glm::max(glm::dot(a, b), 0.f);
    }

    FUNC_QUALIFIER inline float absDot(glm::vec3 a, glm::vec3 b)
    {
        return glm::abs(glm::dot(a, b));
    }

    FUNC_QUALIFIER inline float pow5(float x)
    {
        float x2 = x * x;
        return x2 * x2 * x;
    }

    FUNC_QUALIFIER inline float square(float x)
    {
        return x * x;
    }

    FUNC_QUALIFIER inline float powerHeuristic(float f, float g)
    {
        float f2 = f * f;
        return f2 / (f2 + g * g);
    }

    FUNC_QUALIFIER inline float triangleArea(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
    {
        return glm::length(glm::cross(v1 - v0, v2 - v0)) * .5f;
    }

    FUNC_QUALIFIER inline glm::vec3 triangleNormal(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
    {
        return glm::normalize(glm::cross(v1 - v0, v2 - v0));
    }

    FUNC_QUALIFIER static glm::vec3 sampleTriangleUniform(
        glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float ru, float rv)
    {
        float r = glm::sqrt(rv);
        float u = 1.f - r;
        float v = ru * r;
        return v1 * u + v2 * v + v0 * (1.f - u - v);
    }

    template <typename T>
    FUNC_QUALIFIER inline T calcFilmic(T c)
    {
        return (c * (c * 0.22f + 0.03f) + 0.002f) / (c * (c * 0.22f + 0.3f) + 0.06f) - 1.f / 30.f;
    }

    FUNC_QUALIFIER inline glm::vec3 filmic(glm::vec3 c)
    {
        return calcFilmic(c * 1.6f) / calcFilmic(11.2f);
    }

    FUNC_QUALIFIER inline glm::vec3 ACES(glm::vec3 color)
    {
        return (color * (color * 2.51f + 0.03f)) / (color * (color * 2.43f + 0.59f) + 0.14f);
    }

    FUNC_QUALIFIER inline glm::vec3 correctGamma(glm::vec3 color)
    {
        return glm::pow(color, glm::vec3(1.f / 2.2f));
    }

    FUNC_QUALIFIER inline float luminance(glm::vec3 color)
    {
        // const glm::vec3 T(.299f, .587f, .114f);
        const glm::vec3 T(.2126f, .7152f, .0722f);
        return glm::dot(color, T);
    }

    /**
     * Map a pair of evenly distributed [0, 1] coordinate to disc
     */
    FUNC_QUALIFIER static glm::vec2 toConcentricDisk(float x, float y)
    {
        float r = glm::sqrt(x);
        float theta = y * Pi * 2.0f;
        return glm::vec2(glm::cos(theta), glm::sin(theta)) * r;
    }

    FUNC_QUALIFIER static glm::vec3 toSphere(glm::vec2 v)
    {
        v *= glm::vec2(PiTwo, Pi);
        return glm::vec3(glm::cos(v.x) * glm::sin(v.y), glm::cos(v.y), glm::sin(v.x) * glm::sin(v.y));
    }

    FUNC_QUALIFIER static glm::vec2 toPlane(glm::vec3 v)
    {
        return glm::vec2(
            glm::fract(glm::atan(v.z, v.x) * PiInv * .5f + 1.f),
            glm::atan(glm::length(glm::vec2(v.x, v.z)), v.y) * PiInv);
    }

    FUNC_QUALIFIER static glm::mat3 localRefMatrix(glm::vec3 n)
    {
        glm::vec3 t = (glm::abs(n.y) > 0.9999f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        glm::vec3 b = glm::normalize(glm::cross(n, t));
        t = glm::cross(b, n);
        return glm::mat3(t, b, n);
    }

    FUNC_QUALIFIER static glm::vec3 localToWorld(glm::vec3 n, glm::vec3 v)
    {
        return glm::normalize(localRefMatrix(n) * v);
    }

    FUNC_QUALIFIER static glm::vec3 sampleHemisphereCosine(glm::vec3 n, float rx, float ry)
    {
        glm::vec2 d = toConcentricDisk(rx, ry);
        float z = glm::sqrt(1.f - glm::dot(d, d));
        return localToWorld(n, glm::vec3(d, z));
    }

    FUNC_QUALIFIER static bool refract(glm::vec3 n, glm::vec3 wi, float ior, glm::vec3 &wt)
    {
        float cosIn = glm::dot(n, wi);
        if (cosIn < 0)
        {
            ior = 1.f / ior;
        }
        float sin2In = glm::max(0.f, 1.f - cosIn * cosIn);
        float sin2Tr = sin2In / (ior * ior);

        if (sin2Tr >= 1.f)
        {
            return false;
        }
        float cosTr = glm::sqrt(1.f - sin2Tr);
        if (cosIn < 0)
        {
            cosTr = -cosTr;
        }
        wt = glm::normalize(-wi / ior + n * (cosIn / ior - cosTr));
        return true;
    }

    FUNC_QUALIFIER inline float pdfAreaToSolidAngle(float pdf, glm::vec3 x, glm::vec3 y, glm::vec3 ny)
    {
        glm::vec3 yx = x - y;
        return pdf * glm::dot(yx, yx) / absDot(ny, glm::normalize(yx));
    }

    /**
     * Handy-dandy hash function that provides seeds for random number generation.
     */
    FUNC_QUALIFIER inline unsigned int utilhash(unsigned int a)
    {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }
}