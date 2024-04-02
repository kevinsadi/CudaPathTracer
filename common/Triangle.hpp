#pragma once

#include "BVH.hpp"
#include "Intersection.hpp"
#include "Material.hpp"
#include "Object.hpp"
#include "global.hpp"
#include <cassert>
#include <array>

__host__ __device__ static bool rayTriangleIntersect(const Vector3f& v0, const Vector3f& v1,
    const Vector3f& v2, const Vector3f& orig,
    const Vector3f& dir, float& tnear, float& u, float& v)
{
    Vector3f edge1 = v1 - v0;
    Vector3f edge2 = v2 - v0;
    Vector3f pvec = crossProduct(dir, edge2);
    float det = dotProduct(edge1, pvec);
    if (det == 0 || det < 0)
        return false;

    Vector3f tvec = orig - v0;
    u = dotProduct(tvec, pvec);
    if (u < 0 || u > det)
        return false;

    Vector3f qvec = crossProduct(tvec, edge1);
    v = dotProduct(dir, qvec);
    if (v < 0 || u + v > det)
        return false;

    float invDet = 1 / det;

    tnear = dotProduct(edge2, qvec) * invDet;
    u *= invDet;
    v *= invDet;

    return true;
}

class Triangle : public Object
{
public:
    Vector3f v0, v1, v2; // vertices A, B ,C , counter-clockwise order
    Vector3f e1, e2;     // 2 edges v1-v0, v2-v0;
    Vector3f t0, t1, t2; // texture coords
    Vector3f normal;
    float area = -1;
    Material* material = nullptr;

    Triangle() = default;

    __host__ __device__ Triangle(Vector3f _v0, Vector3f _v1, Vector3f _v2, Material* _m = nullptr)
        : v0(_v0), v1(_v1), v2(_v2), material(_m)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = normalize(crossProduct(e1, e2));
        area = crossProduct(e1, e2).norm() * 0.5f;
    }

    __host__ __device__ bool intersect(const Ray& ray) override;
    __host__ __device__ bool intersect(const Ray& ray, float& tnear,
        uint32_t& index) const override;
    __host__ __device__ Intersection getIntersection(Ray ray) override;
    __host__ __device__ void getSurfaceProperties(const Vector3f& P, const Vector3f& I,
        const uint32_t& index, const Vector2f& uv,
        Vector3f& N, Vector2f& st) const override
    {
        N = normal;
        //        throw std::runtime_error("triangle::getSurfaceProperties not
        //        implemented.");
    }
    __host__ __device__ Vector3f evalDiffuseColor(const Vector2f&) const override;
    __host__ __device__ Bounds3 getBounds() override;
    // // Sample a point on the surface of the object, used for area light
    // __host__ __device__ void Sample(Intersection& pos, float& pdf) override {
    //     float x = std::sqrt(get_random_float()), y = get_random_float();
    //     pos.coords = v0 * (1.0f - x) + v1 * (x * (1.0f - y)) + v2 * (x * y);
    //     pos.normal = this->normal;
    //     pdf = 1.0f / area;
    // }
    __host__ __device__ float getArea() override {
        return area;
    }
    __host__ __device__ bool hasEmit() override {
        return material->hasEmission();
    }

    CUDA_PORTABLE(Triangle);
};

class MeshTriangle : public Object
{
public:
    MeshTriangle(const std::string& filename, Material* mt = new Material());
    ~MeshTriangle();

    __host__ __device__ bool intersect(const Ray& ray) override { return true; }

    __host__ __device__ bool intersect(const Ray& ray, float& tnear, uint32_t& index) const override
    {
        bool intersect = false;
        for (uint32_t k = 0; k < numTriangles; ++k) {
            const Vector3f& v0 = vertices[vertexIndex[k * 3]];
            const Vector3f& v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vector3f& v2 = vertices[vertexIndex[k * 3 + 2]];
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

    __host__ __device__ Bounds3 getBounds() override { return bounding_box; }

    __host__ __device__ void getSurfaceProperties(const Vector3f& P, const Vector3f& I,
        const uint32_t& index, const Vector2f& uv,
        Vector3f& N, Vector2f& st) const override
    {
        const Vector3f& v0 = vertices[vertexIndex[index * 3]];
        const Vector3f& v1 = vertices[vertexIndex[index * 3 + 1]];
        const Vector3f& v2 = vertices[vertexIndex[index * 3 + 2]];
        Vector3f e0 = normalize(v1 - v0);
        Vector3f e1 = normalize(v2 - v1);
        N = normalize(crossProduct(e0, e1));
        const Vector2f& st0 = stCoordinates[vertexIndex[index * 3]];
        const Vector2f& st1 = stCoordinates[vertexIndex[index * 3 + 1]];
        const Vector2f& st2 = stCoordinates[vertexIndex[index * 3 + 2]];
        st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
    }

    __host__ __device__ Vector3f evalDiffuseColor(const Vector2f& st) const override
    {
        float scale = 5;
        float pattern =
            (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
        return lerp(Vector3f(0.815, 0.235, 0.031),
            Vector3f(0.937, 0.937, 0.231), pattern);
    }

    __host__ __device__ Intersection getIntersection(Ray ray) override
    {
        Intersection intersec;

        if (bvh) {
            intersec = bvh->Intersect(ray);
        }

        return intersec;
    }
    // Sample a point on the surface of the object, used for area light
    __host__ __device__ void Sample(Intersection& pos, float& pdf) override {
        bvh->Sample(pos, pdf);
        pos.emit = material->getEmission();
    }
    __host__ __device__ float getArea() override {
        return area;
    }
    __host__ __device__ bool hasEmit() override {
        return material->hasEmission();
    }

    Bounds3 bounding_box;
    // todo: we have both triangle soup and indexed mesh, should keep only one
    int num_vertices = 0;
    Vector3f* vertices = nullptr;
    uint32_t numTriangles;
    Vector2f* stCoordinates = nullptr; // uv1
    // todo: vertex normal?
    int num_triangles = 0;
    uint32_t* vertexIndex = nullptr;
    Triangle* triangles = nullptr; // triangle soup

    BVHAccel* bvh;
    float area;

    Material* material;

    CUDA_PORTABLE(MeshTriangle);
};

__host__ __device__ inline bool Triangle::intersect(const Ray& ray) { return true; }
__host__ __device__ inline bool Triangle::intersect(const Ray& ray, float& tnear,
    uint32_t& index) const
{
    return false;
}

__host__ __device__ inline Bounds3 Triangle::getBounds() { return Union(Bounds3(v0, v1), v2); }

__host__ __device__ inline Intersection Triangle::getIntersection(Ray ray)
{
    Intersection inter;

    if (dotProduct(ray.direction, normal) > 0)
        return inter;
    double u, v, t_tmp = 0;
    Vector3f pvec = crossProduct(ray.direction, e2);
    double det = dotProduct(e1, pvec);
    if (fabs(det) < Epsilon)
        return inter;

    double det_inv = 1. / det;
    Vector3f tvec = ray.origin - v0;
    u = dotProduct(tvec, pvec) * det_inv;
    if (u < 0 || u > 1)
        return inter;
    Vector3f qvec = crossProduct(tvec, e1);
    v = dotProduct(ray.direction, qvec) * det_inv;
    if (v < 0 || u + v > 1)
        return inter;
    t_tmp = dotProduct(e2, qvec) * det_inv;

    if (t_tmp < 0)
        return inter;

    inter.happened = true;
    inter.coords = ray(t_tmp);
    inter.normal = normal;
    inter.distance = t_tmp;
    inter.obj = this;
    inter.m = this->material;

    return inter;
}

__host__ __device__ inline Vector3f Triangle::evalDiffuseColor(const Vector2f&) const
{
    return Vector3f(0.5, 0.5, 0.5);
}
