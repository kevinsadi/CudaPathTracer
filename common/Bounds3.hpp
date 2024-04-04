//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BOUNDS3_H
#define RAYTRACING_BOUNDS3_H
#include "Ray.hpp"
#include "MathUtils.hpp"
#include <limits>
#include <array>

class Bounds3
{
public:
    glm::vec3 pMin, pMax; // two points to specify the bounding box
    FUNC_QUALIFIER inline Bounds3()
    {
        double minNum = kDoubleNegInfinity;
        double maxNum = kDoubleInfinity;
        pMax = glm::vec3(minNum, minNum, minNum);
        pMin = glm::vec3(maxNum, maxNum, maxNum);
    }
    FUNC_QUALIFIER inline Bounds3(const glm::vec3 p) : pMin(p), pMax(p) {}
    FUNC_QUALIFIER inline Bounds3(const glm::vec3 p1, const glm::vec3 p2)
    {
        pMin = glm::vec3(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
        pMax = glm::vec3(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
    }

    FUNC_QUALIFIER inline glm::vec3 Diagonal() const { return pMax - pMin; }
    FUNC_QUALIFIER inline int maxExtent() const
    {
        glm::vec3 d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    FUNC_QUALIFIER inline double SurfaceArea() const
    {
        glm::vec3 d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    FUNC_QUALIFIER inline glm::vec3 Centroid() { return pMin * 0.5f + pMax * 0.5f; }
    FUNC_QUALIFIER inline Bounds3 Intersect(const Bounds3& b)
    {
        return Bounds3(glm::vec3(glm::max(pMin.x, b.pMin.x), glm::max(pMin.y, b.pMin.y),
            glm::max(pMin.z, b.pMin.z)),
            glm::vec3(glm::min(pMax.x, b.pMax.x), glm::min(pMax.y, b.pMax.y),
                glm::min(pMax.z, b.pMax.z)));
    }

    FUNC_QUALIFIER inline glm::vec3 Offset(const glm::vec3& p) const
    {
        glm::vec3 o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    FUNC_QUALIFIER inline bool Overlaps(const Bounds3& b1, const Bounds3& b2)
    {
        bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
        bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
        bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
        return (x && y && z);
    }

    FUNC_QUALIFIER inline bool Inside(const glm::vec3& p, const Bounds3& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
            p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
    }
    FUNC_QUALIFIER inline const glm::vec3& operator[](int i) const
    {
        return (i == 0) ? pMin : pMax;
    }

    FUNC_QUALIFIER inline bool IntersectP(const Ray& ray, const glm::vec3& invDir,
        const int* dirisNeg) const;
};



FUNC_QUALIFIER inline bool Bounds3::IntersectP(const Ray& ray, const glm::vec3& invDir,
    const int* dirIsNeg) const
{
    // dirIsNeg[0] = invDir.x < 0;
    double tMin = ((dirIsNeg[0] ? pMax.x : pMin.x) - ray.origin.x) * invDir.x;
    double tMax = (((1 - dirIsNeg[0]) ? pMax.x : pMin.x) - ray.origin.x) * invDir.x;
    double tyMin = ((dirIsNeg[1] ? pMax.y : pMin.y) - ray.origin.y) * invDir.y;
    double tyMax = (((1 - dirIsNeg[1]) ? pMax.y : pMin.y) - ray.origin.y) * invDir.y;
    // checking x and y axis
    if ((tMin > tyMax) || (tyMin > tMax))
        return false;
    if (tyMin > tMin)
        tMin = tyMin;
    if (tyMax < tMax)
        tMax = tyMax;
    double tzMin = ((dirIsNeg[2] ? pMax.z : pMin.z) - ray.origin.z) * invDir.z;
    double tzMax = (((1 - dirIsNeg[2]) ? pMax.z : pMin.z) - ray.origin.z) * invDir.z;
    if ((tMin > tzMax) || (tzMin > tMax))
        return false;
    // checking z axis
    if (tzMin > tMin)
        tMin = tzMin;
    if (tzMax < tMax)
        tMax = tzMax;
    return true;
}

FUNC_QUALIFIER inline Bounds3 Union(const Bounds3& b1, const Bounds3& b2)
{
    Bounds3 ret;
    ret.pMin = glm::min(b1.pMin, b2.pMin);
    ret.pMax = glm::max(b1.pMax, b2.pMax);
    return ret;
}

FUNC_QUALIFIER inline Bounds3 Union(const Bounds3& b, const glm::vec3& p)
{
    Bounds3 ret;
    ret.pMin = glm::min(b.pMin, p);
    ret.pMax = glm::max(b.pMax, p);
    return ret;
}

#endif // RAYTRACING_BOUNDS3_H
