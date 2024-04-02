//
// Created by LEI XU on 5/13/19.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include "Object.hpp"
#include "Vector.hpp"
#include "Bounds3.hpp"
#include "Material.hpp"
#include "MathUtils.hpp"

class Sphere : public Object {
public:
    Vector3f center;
    float radius, radius2;
    Material* material;
    float area;
    Sphere(const Vector3f& c, const float& r, Material* mt = new Material()) : center(c), radius(r), radius2(r* r), material(mt), area(4 * Pi * r * r) {}
    FUNC_QUALIFIER bool intersect(const Ray& ray) {
        // analytic solution
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        float area = 4 * Pi * radius2;
        if (!solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }
    FUNC_QUALIFIER bool intersect(const Ray& ray, float& tnear, uint32_t& index) const
    {
        // analytic solution
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!
            solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        tnear = t0;

        return true;
    }
    FUNC_QUALIFIER Intersection getIntersection(Ray ray) {
        Intersection result;
        result.happened = false;
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return result;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return result;
        result.happened = true;

        result.coords = Vector3f(ray.origin + ray.direction * t0);
        result.normal = normalize(Vector3f(result.coords - center));
        result.m = this->material;
        result.obj = this;
        result.distance = t0;
        return result;

    }
    FUNC_QUALIFIER void getSurfaceProperties(const Vector3f& P, const Vector3f& I, const uint32_t& index, const Vector2f& uv, Vector3f& N, Vector2f& st) const
    {
        N = normalize(P - center);
    }

    FUNC_QUALIFIER Vector3f evalDiffuseColor(const Vector2f& st) const {
        //return m->getColor();
        return material->m_albedo;
    }
    FUNC_QUALIFIER Bounds3 getBounds() {
        return Bounds3(Vector3f(center.x - radius, center.y - radius, center.z - radius),
            Vector3f(center.x + radius, center.y + radius, center.z + radius));
    }
    // sample a point on the surface of the sphere, used for area light
    FUNC_QUALIFIER void Sample(Intersection& pos, float& pdf) {
        float theta = 2.0 * Pi * get_random_float(), phi = Pi * get_random_float();
        Vector3f dir(glm::cos(phi), glm::sin(phi) * glm::cos(theta), glm::sin(phi) * glm::sin(theta));
        pos.coords = center + radius * dir;
        pos.normal = dir;
        pos.emit = material->getEmission();
        pdf = 1.0f / area;
    }
    FUNC_QUALIFIER float getArea() {
        return area;
    }
    FUNC_QUALIFIER bool hasEmit() {
        return material->hasEmission();
    }
    CUDA_PORTABLE(Sphere);
};




#endif //RAYTRACING_SPHERE_H
