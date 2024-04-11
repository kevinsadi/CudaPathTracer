#pragma once

#include "CudaPortable.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef GPU_PATH_TRACER
#include <thrust/random.h>
using Sampler = thrust::default_random_engine;
using Distribution = thrust::uniform_real_distribution<float>;
#else
#include <random>
using Sampler = std::mt19937;
using Distribution = std::uniform_real_distribution<float>;
#endif

struct RNG {
    Sampler sampler;
    FUNC_QUALIFIER inline RNG(int iter, int index, int dim, uint32_t *data) {
        int h = utilhash((1 << 31) | (dim << 22) | iter) ^ utilhash(index);
        sampler = Sampler(h);
    }

    FUNC_QUALIFIER inline float get_random_float() {
        return Distribution(0.f, 1.f)(sampler);
    }

    FUNC_QUALIFIER inline float sample1D() {
        return get_random_float();
    }

    FUNC_QUALIFIER inline glm::vec2 sample2D() {
        return glm::vec2(get_random_float(), get_random_float());
    }

    FUNC_QUALIFIER inline glm::vec3 sample3D() {
        return glm::vec3(sample2D(), get_random_float());
    }

    FUNC_QUALIFIER inline glm::vec4 sample4D() {
        return glm::vec4(sample2D(), sample2D());
    }

    /**
     * Handy-dandy hash function that provides seeds for random number generation.
     */
    FUNC_QUALIFIER inline static unsigned int utilhash(unsigned int a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }
};

#define Pi 3.1415926535897932384626422832795028841971f
#define PiTwo 6.2831853071795864769252867665590057683943f
#define PiInv 1.f / Pi
#define Epsilon5 1e-5f
#define Epsilon8 1e-8f
#define Epsilon4 (5 * 1e-4f)

#define kDoubleInfinity 1e30f
#define kDoubleNegInfinity -kDoubleInfinity
#define kFloatInfinity 3.402823466e+38F
#define kFloatNegInfinity -kFloatInfinity

FUNC_QUALIFIER inline void swap(float &a, float &b) {
    float temp = a;
    a = b;
    b = temp;
}

FUNC_QUALIFIER inline float clamp(const float &lo, const float &hi, const float &v) {
    return glm::max(lo, glm::min(hi, v));
}

FUNC_QUALIFIER inline bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1) {
    float discr = b * b - 4 * a * c;
    if (discr < 0)
        return false;
    else if (discr == 0)
        x0 = x1 = -0.5 * b / a;
    else {
        float q = (b > 0) ? -0.5 * (b + sqrt(discr)) : -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1)
        swap(x0, x1);
    return true;
}

static std::string vec3ToString(const glm::vec3 &vec) {
    std::stringstream ss;
    ss << "{ x = " << vec.x << ", y = " << vec.y << ", z = " << vec.z << " }";
    return ss.str();
}

namespace Math {
    FUNC_QUALIFIER inline glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
        glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
        glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * Pi / 180.f, glm::vec3(1.f, 0.f, 0.f));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * Pi / 180.f, glm::vec3(0.f, 1.f, 0.f));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * Pi / 180.f, glm::vec3(0.f, 0.f, 1.f));
        glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
        return translationMat * rotationMat * scaleMat;
    }
    template <typename T>
    FUNC_QUALIFIER inline bool between(const T &x, const T &min, const T &max) {
        return x >= min && x <= max;
    }

    // given a local hemisphere direction, and a global normal corresponding to the Hemisphere axis
    // return the world space direction
    FUNC_QUALIFIER inline glm::vec3 local_to_world(const glm::vec3 &local_dir, const glm::vec3 &normal) {
        glm::vec3 t;
        if (glm::abs(normal.x) > glm::abs(normal.y)) {
            const auto inv_len = 1.0f / glm::sqrt(normal.x * normal.x + normal.z * normal.z);
            t = {normal.z * inv_len, 0.0f, -normal.x * inv_len};
        } else {
            const auto inv_len = 1.0f / glm::sqrt(normal.y * normal.y + normal.z * normal.z);
            t = {0.0f, normal.z * inv_len, -normal.y * inv_len};
        }
        const auto b = glm::cross(t, normal);
        return local_dir.x * b + local_dir.y * t + local_dir.z * normal;
    }
    // transform polar coordinates to cartesian coordinates
    FUNC_QUALIFIER inline glm::vec3 polar_to_cartesian(float theta, float phi) {
        return {
            glm::sin(theta) * glm::cos(phi),
            glm::sin(theta) * glm::sin(phi),
            glm::cos(theta)};
    }

    FUNC_QUALIFIER inline float pow5(float x) {
        float x2 = x * x;
        return x2 * x2 * x;
    }

    FUNC_QUALIFIER inline float lerp(float x, float y, float t) {
        return x * (1.0f - t) + y * t;
    }
    FUNC_QUALIFIER inline glm::vec3 lerp(const glm::vec3 &a, const glm::vec3 &b, float t) {
        return a * (1.0f - t) + b * t;
    }
    FUNC_QUALIFIER inline float length2(const glm::vec3 &v) {
        return glm::dot(v, v);
    }

    FUNC_QUALIFIER inline float satDot(glm::vec3 a, glm::vec3 b) {
        return glm::max(glm::dot(a, b), 0.f);
    }

    FUNC_QUALIFIER inline float absDot(glm::vec3 a, glm::vec3 b) {
        return glm::abs(glm::dot(a, b));
    }
    FUNC_QUALIFIER inline float square(float x) {
        return x * x;
    }
}