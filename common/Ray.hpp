//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_RAY_H
#define RAYTRACING_RAY_H
#include "MathUtils.hpp"
struct Ray
{
    // Destination = origin + t*direction
    glm::vec3 origin;
    glm::vec3 direction, direction_inv;
    double t; // transportation time,
    double t_min, t_max;

    FUNC_QUALIFIER inline Ray(const glm::vec3 &ori, const glm::vec3 &dir, const double _t = 0.0) : origin(ori), direction(dir), t(_t)
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
#endif // RAYTRACING_RAY_H
