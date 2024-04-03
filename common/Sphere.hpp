//
// Created by LEI XU on 5/13/19.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include "Object.hpp"
#include "Bounds3.hpp"
#include "Material.hpp"
#include "MathUtils.hpp"

class Sphere : public Object {
public:
    glm::vec3 center;
    float radius, radius2;
    Material* material;
    float area;
    Sphere(const glm::vec3& c, const float& r, Material* mt = new Material()) : center(c), radius(r), radius2(r* r), material(mt), area(4 * Pi * r * r) {}
    FUNC_QUALIFIER inline bool intersect(const Ray& ray) {
        // analytic solution
        glm::vec3 L = ray.origin - center;
        float a = glm::dot(ray.direction, ray.direction);
        float b = 2 * glm::dot(ray.direction, L);
        float c = glm::dot(L, L) - radius2;
        float t0, t1;
        float area = 4 * Pi * radius2;
        if (!solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }
    FUNC_QUALIFIER inline bool intersect(const Ray& ray, float& tnear, uint32_t& index) const
    {
        // analytic solution
        glm::vec3 L = ray.origin - center;
        float a = glm::dot(ray.direction, ray.direction);
        float b = 2 * glm::dot(ray.direction, L);
        float c = glm::dot(L, L) - radius2;
        float t0, t1;
        if (!
            solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        tnear = t0;

        return true;
    }
    FUNC_QUALIFIER inline Intersection getIntersection(Ray ray) {
        Intersection result;
        result.happened = false;
        glm::vec3 L = ray.origin - center;
        float a = glm::dot(ray.direction, ray.direction);
        float b = 2 * glm::dot(ray.direction, L);
        float c = glm::dot(L, L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return result;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return result;
        result.happened = true;

        result.coords = glm::vec3(ray.origin + ray.direction * t0);
        result.normal = normalize(glm::vec3(result.coords - center));
        result.m = this->material;
        result.obj = this;
        result.distance = t0;
        return result;

    }
    FUNC_QUALIFIER inline void getSurfaceProperties(const glm::vec3& P, const glm::vec3& I, const uint32_t& index, const glm::vec2& uv, glm::vec3& N, glm::vec2& st) const
    {
        N = normalize(P - center);
    }

    FUNC_QUALIFIER inline glm::vec3 evalDiffuseColor(const glm::vec2& st) const {
        //return m->getColor();
        return material->m_albedo;
    }
    FUNC_QUALIFIER inline Bounds3 getBounds() {
        return Bounds3(glm::vec3(center.x - radius, center.y - radius, center.z - radius),
            glm::vec3(center.x + radius, center.y + radius, center.z + radius));
    }
    // sample a point on the surface of the sphere, used for area light
    FUNC_QUALIFIER inline void Sample(Intersection& pos, float& pdf) {
        float theta = 2.0 * Pi * get_random_float(), phi = Pi * get_random_float();
        glm::vec3 dir(glm::cos(phi), glm::sin(phi) * glm::cos(theta), glm::sin(phi) * glm::sin(theta));
        pos.coords = center + radius * dir;
        pos.normal = dir;
        pos.emit = material->getEmission();
        pdf = 1.0f / area;
    }
    FUNC_QUALIFIER inline float getArea() {
        return area;
    }
    FUNC_QUALIFIER inline bool hasEmit() {
        return material->hasEmission();
    }
    CUDA_PORTABLE(Sphere);
};




#endif //RAYTRACING_SPHERE_H
