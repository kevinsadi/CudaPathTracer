//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include "MathUtils.hpp"

enum MaterialType { Lambert, Metal, Dielectric };

class Material {
private:

    // Compute reflection direction
    FUNC_QUALIFIER inline glm::vec3 reflect(const glm::vec3& I, const glm::vec3& N) const
    {
        return I - 2 * glm::dot(I, N) * N;
    }

    // Compute refraction direction using Snell's law
    //
    // We need to handle with care the two possible situations:
    //
    //    - When the ray is inside the object
    //
    //    - When the ray is outside.
    //
    // If the ray is outside, you need to make cosi positive cosi = -N.I
    //
    // If the ray is inside, you need to invert the refractive indices and negate the normal N
    FUNC_QUALIFIER inline glm::vec3 refract(const glm::vec3& I, const glm::vec3& N, const float& ior) const
    {
        // float cosi = clamp(-1, 1, glm::dot(I, N));
        // float etai = 1, etat = ior;
        // glm::vec3 n = N;
        // if (cosi < 0) { cosi = -cosi; }
        // else { swap(etai, etat); n = -N; }
        // float eta = etai / etat;
        // float k = 1 - eta * eta * (1 - cosi * cosi);
        // return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;

        return glm::vec3(0.0f);
    }

    // Compute Fresnel equation
    //
    // \param I is the incident view direction
    //
    // \param N is the normal at the intersection point
    //
    // \param ior is the material refractive index
    //
    // \param[out] kr is the amount of light reflected
    FUNC_QUALIFIER inline void fresnel(const glm::vec3& I, const glm::vec3& N, const float& ior, float& kr) const
    {
        float cosi = clamp(-1, 1, glm::dot(I, N));
        float etai = 1, etat = ior;
        if (cosi > 0) { swap(etai, etat); }
        // Compute sini using Snell's law
        float sint = etai / etat * sqrtf(glm::max(0.f, 1 - cosi * cosi));
        // Total internal reflection
        if (sint >= 1) {
            kr = 1;
        }
        else {
            float cost = sqrtf(glm::max(0.f, 1 - sint * sint));
            cosi = fabsf(cosi);
            float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
            float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
            kr = (Rs * Rs + Rp * Rp) / 2;
        }
        // As a consequence of the conservation of energy, transmittance is given by:
        // kt = 1 - kr;
    }

    FUNC_QUALIFIER inline glm::vec3 toWorld(const glm::vec3& a, const glm::vec3& N) {
        glm::vec3 B, C;
        if (glm::abs(N.x) > glm::abs(N.y)) {
            float invLen = 1.0f / glm::sqrt(N.x * N.x + N.z * N.z);
            C = glm::vec3(N.z * invLen, 0.0f, -N.x * invLen);
        }
        else {
            float invLen = 1.0f / glm::sqrt(N.y * N.y + N.z * N.z);
            C = glm::vec3(0.0f, N.z * invLen, -N.y * invLen);
        }
        B = glm::cross(C, N);
        return a.x * B + a.y * C + a.z * N;
    }

public:
    MaterialType m_type;
    glm::vec3 m_albedo;
    glm::vec3 m_emissive;
    float m_ior, roughness;

    inline Material(MaterialType t = Lambert, glm::vec3 e = glm::vec3(0, 0, 0));
    FUNC_QUALIFIER inline MaterialType getType();
    //inline glm::vec3 getColor();
    FUNC_QUALIFIER inline glm::vec3 getColorAt(double u, double v);
    FUNC_QUALIFIER inline glm::vec3 getEmission();
    FUNC_QUALIFIER inline bool hasEmission();

    // sample a ray by Material properties
    FUNC_QUALIFIER inline glm::vec3 sample(const glm::vec3& wi, const glm::vec3& N);
    // given a ray, calculate the PdF of this ray
    FUNC_QUALIFIER inline float pdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& N);
    // given a ray, calculate the contribution of this ray
    FUNC_QUALIFIER inline glm::vec3 eval(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& N);

    CUDA_PORTABLE(Material);
};

Material::Material(MaterialType t, glm::vec3 e) : m_type(t), m_emissive(e) {}

MaterialType Material::getType() { return m_type; }
///glm::vec3 Material::getColor(){return m_color;}
glm::vec3 Material::getEmission() { return m_emissive; }
bool Material::hasEmission() {
    if (glm::length(m_emissive) > Epsilon) return true;
    else return false;
}

glm::vec3 Material::getColorAt(double u, double v) {
    return glm::vec3();
}


glm::vec3 Material::sample(const glm::vec3& wi, const glm::vec3& N) {
    switch (m_type) {
    case Lambert:
    {
        // uniform sample on the hemisphere
        float x_1 = get_random_float(), x_2 = get_random_float();
        float z = glm::abs(1.0f - 2.0f * x_1);
        float r = glm::sqrt(1.0f - z * z), phi = 2 * Pi * x_2;
        glm::vec3 localRay(r * glm::cos(phi), r * glm::sin(phi), z);
        return toWorld(localRay, N);

        break;
    }
    case Metal:
    case Dielectric:
    {
        return glm::vec3(0.0f);
        break;
    }
    }
}

float Material::pdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& N) {
    switch (m_type) {
    case Lambert:
    {
        // uniform sample probability 1 / (2 * PI)
        if (glm::dot(wo, N) > 0.0f)
            return 0.5f * PiInv;
        else
            return 0.0f;
        break;
    }
    case Metal:
    case Dielectric:
    {
        return 0.0f;
        break;
    }
    }
}

glm::vec3 Material::eval(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& N) {
    switch (m_type) {
    case Lambert:
    {
        // calculate the contribution of diffuse   model
        float cosalpha = glm::dot(N, wo);
        if (cosalpha > 0.0f) {
            glm::vec3 diffuse = m_albedo * PiInv;
            return diffuse;
        }
        else
            return glm::vec3(0.0f);
        break;
    }
    case Metal:
    case Dielectric:
    {
        return glm::vec3(0.0f);
        break;
    }
    }
}

#endif //RAYTRACING_MATERIAL_H
