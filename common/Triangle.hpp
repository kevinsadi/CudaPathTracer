#pragma once

// #include "BVH.hpp"
#include "Intersection.hpp"
#include "Material.hpp"
#include "Object.hpp"
#include "MathUtils.hpp"
#include "Ray.hpp"
#include <cassert>
#include <array>

FUNC_QUALIFIER inline static bool rayTriangleIntersect(const glm::vec3& v0, const glm::vec3& v1,
    const glm::vec3& v2, const glm::vec3& orig,
    const glm::vec3& dir, float& tnear, float& u, float& v)
{
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 pvec = glm::cross(dir, edge2);
    float det = glm::dot(edge1, pvec);
    if (det == 0 || det < 0)
        return false;

    glm::vec3 tvec = orig - v0;
    u = glm::dot(tvec, pvec);
    if (u < 0 || u > det)
        return false;

    glm::vec3 qvec = glm::cross(tvec, edge1);
    v = glm::dot(dir, qvec);
    if (v < 0 || u + v > det)
        return false;

    float invDet = 1 / det;

    tnear = glm::dot(edge2, qvec) * invDet;
    u *= invDet;
    v *= invDet;

    return true;
}

class Triangle : public Object
{
public:
    glm::vec3 v0, v1, v2; // vertices A, B ,C , counter-clockwise order
    glm::vec3 e1, e2;     // 2 edges v1-v0, v2-v0;
    glm::vec3 t0, t1, t2; // texture coords
    glm::vec3 normal;
    Material material;

    FUNC_QUALIFIER inline Triangle() {}
    FUNC_QUALIFIER inline Triangle(glm::vec3 _v0, glm::vec3 _v1, glm::vec3 _v2, Material& _m)
        : v0(_v0), v1(_v1), v2(_v2), material(_m)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = normalize(glm::cross(e1, e2));
        area = glm::length(glm::cross(e1, e2)) * 0.5f;
    }

    FUNC_QUALIFIER inline bool intersect(const Ray& ray, float& tnear,
        uint32_t& index) const override;
    FUNC_QUALIFIER inline Intersection getIntersection(Ray ray);
    // FUNC_QUALIFIER inline void getSurfaceProperties(const glm::vec3& P, const glm::vec3& I,
    //     const uint32_t& index, const glm::vec2& uv,
    //     glm::vec3& N, glm::vec2& st) const override
    // {
    //     N = normal;
    //     //        throw std::runtime_error("triangle::getSurfaceProperties not
    //     //        implemented.");
    // }
    // FUNC_QUALIFIER inline glm::vec3 evalDiffuseColor(const glm::vec2&) const override;
    FUNC_QUALIFIER inline Bounds3 getBounds() override;
    // Sample a point on the surface of the object, used for area light
    FUNC_QUALIFIER inline void Sample(RNG& rng, Intersection& pos, float& pdf) {
        float x = glm::sqrt(rng.sample1D()), y = rng.sample1D();
        pos.coords = v0 * (1.0f - x) + v1 * (x * (1.0f - y)) + v2 * (x * y);
        pos.normal = this->normal;
        pos.triangleArea = this->area;
        // pdf = 1.0f / area;
        // pdf = 1.0f;
    }

    CUDA_PORTABLE(Triangle);
};

class MeshTriangle : public Object
{
public:
    MeshTriangle(const std::string& filename, Material& mt);
    ~MeshTriangle();


    FUNC_QUALIFIER inline bool intersect(const Ray& ray, float& tnear, uint32_t& index) const override
    {
        bool intersect = false;
        for (uint32_t k = 0; k < num_triangles; ++k) {
            const glm::vec3& v0 = vertices[vertexIndex[k * 3]];
            const glm::vec3& v1 = vertices[vertexIndex[k * 3 + 1]];
            const glm::vec3& v2 = vertices[vertexIndex[k * 3 + 2]];
            float t, u, v;
            if (rayTriangleIntersect(v0, v1, v2, ray.origin, ray.direction, t,
                u, v) &&
                t < tnear) {
                tnear = t;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

    FUNC_QUALIFIER inline Bounds3 getBounds() override { return bounding_box; }

    // FUNC_QUALIFIER inline void getSurfaceProperties(const glm::vec3& P, const glm::vec3& I,
    //     const uint32_t& index, const glm::vec2& uv,
    //     glm::vec3& N, glm::vec2& st) const override
    // {
    //     const glm::vec3& v0 = vertices[vertexIndex[index * 3]];
    //     const glm::vec3& v1 = vertices[vertexIndex[index * 3 + 1]];
    //     const glm::vec3& v2 = vertices[vertexIndex[index * 3 + 2]];
    //     glm::vec3 e0 = normalize(v1 - v0);
    //     glm::vec3 e1 = normalize(v2 - v1);
    //     N = normalize(glm::cross(e0, e1));
    //     const glm::vec2& st0 = stCoordinates[vertexIndex[index * 3]];
    //     const glm::vec2& st1 = stCoordinates[vertexIndex[index * 3 + 1]];
    //     const glm::vec2& st2 = stCoordinates[vertexIndex[index * 3 + 2]];
    //     st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
    // }

    // FUNC_QUALIFIER inline glm::vec3 evalDiffuseColor(const glm::vec2& st) const override
    // {
    //     float scale = 5;
    //     float pattern =
    //         (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
    //     return glm::mix(glm::vec3(0.815, 0.235, 0.031),
    //         glm::vec3(0.937, 0.937, 0.231), pattern);
    // }

    Bounds3 bounding_box;
    // todo: we have both triangle soup and indexed mesh, should keep only one
    int num_vertices = 0;
    glm::vec3* vertices = nullptr;
    glm::vec2* stCoordinates = nullptr; // uv1
    // todo: vertex normal?
    int num_triangles = 0;
    uint32_t* vertexIndex = nullptr;
    Triangle* triangles = nullptr; // triangle soup

    Material material;

    CUDA_PORTABLE(MeshTriangle);
};

FUNC_QUALIFIER inline bool Triangle::intersect(const Ray& ray, float& tnear,
    uint32_t& index) const
{
    return false;
}

FUNC_QUALIFIER inline Bounds3 Triangle::getBounds() { return Union(Bounds3(v0, v1), v2); }

FUNC_QUALIFIER inline Intersection Triangle::getIntersection(Ray ray)
{
    Intersection inter;

    if (glm::dot(ray.direction, normal) > 0)
        return inter;
    double u, v, t_tmp = 0;
    glm::vec3 pvec = glm::cross(ray.direction, e2);
    double det = glm::dot(e1, pvec);
    if (fabs(det) < Epsilon5)
        return inter;

    double det_inv = 1. / det;
    glm::vec3 tvec = ray.origin - v0;
    u = glm::dot(tvec, pvec) * det_inv;
    if (u < 0 || u > 1)
        return inter;
    glm::vec3 qvec = glm::cross(tvec, e1);
    v = glm::dot(ray.direction, qvec) * det_inv;
    if (v < 0 || u + v > 1)
        return inter;
    t_tmp = glm::dot(e2, qvec) * det_inv;

    if (t_tmp < 0)
        return inter;

    inter.happened = true;
    inter.coords = ray(t_tmp);
    inter.normal = normal;
    inter.distance = t_tmp;
    inter.triangleArea = area;
    inter.m = this->material;

    return inter;
}

// FUNC_QUALIFIER inline glm::vec3 Triangle::evalDiffuseColor(const glm::vec2&) const
// {
//     return glm::vec3(0.5, 0.5, 0.5);
// }
