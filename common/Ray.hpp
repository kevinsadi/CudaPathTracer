//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_RAY_H
#define RAYTRACING_RAY_H
#include "MathUtils.hpp"
#include "CudaPortable.hpp"
struct Ray
{
    // Destination = origin + t*direction
    glm::vec3 origin;
    glm::vec3 direction, direction_inv;
    double t; // transportation time,
    double t_min, t_max;
    FUNC_QUALIFIER inline Ray() {}
    FUNC_QUALIFIER inline Ray(const glm::vec3& ori, const glm::vec3& dir, const double _t = 0.0) : origin(ori), direction(dir), t(_t)
    {
        direction_inv = glm::vec3(1. / direction.x, 1. / direction.y, 1. / direction.z);
        t_min = 0.0;
        t_max = kDoubleInfinity;
    }

    FUNC_QUALIFIER inline glm::vec3 operator()(double t) const { return origin + direction * (float)t; }

    // friend std::ostream &operator<<(std::ostream &os, const Ray &r)
    // {
    //     os << "[origin:=" << r.origin << ", direction=" << r.direction << ", time=" << r.t << "]\n";
    //     return os;
    // }
};

// struct PathSegment
// {
//     PathSegment() = default;
//     FUNC_QUALIFIER inline PathSegment(const RNG& rng, const Ray& ray, const glm::vec3& throughput, const glm::vec3& radiance, int pixelIndex, int remainingBounces)
//         : rng(rng), ray(ray), throughput(throughput), radiance(radiance), pixelIndex(pixelIndex), remainingBounces(remainingBounces) {}
//     RNG rng;
//     Ray ray;
//     // path states
//     glm::vec3 throughput;
//     glm::vec3 radiance;
//     float bsdfSamplePdf = 0.0f;
//     bool specularBounce = false;
//     bool anyNonSpecularBounces = false;
//     int pixelIndex;
//     int remainingBounces; // if == 0, then the path is terminated
// }
#endif // RAYTRACING_RAY_H
