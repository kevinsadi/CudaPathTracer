#pragma once

#include "CudaPortable.hpp"
#include "MathUtils.hpp"

FUNC_QUALIFIER static inline glm::vec3 reflect(const glm::vec3 &ray_in_dir, const glm::vec3 &normal) {
    return ray_in_dir - glm::dot(ray_in_dir, normal) * normal * 2.f;
}
// If the ray is outside, make cos_i positive.
// If the ray is inside, invert the refractive indices and negate the normal.
FUNC_QUALIFIER static inline glm::vec3 refract(const glm::vec3 &ray_in_dir, const glm::vec3 &normal, float ior) {
    auto cos_i = clamp(-1.0f, 1.0f, glm::dot(ray_in_dir, normal));
    auto eta_i = 1.0f;
    auto eta_t = ior;
    glm::vec3 correct_normal = normal;
    if (cos_i < 0.0f) {
        cos_i = -cos_i;
    } else {
        swap(eta_i, eta_t);
        correct_normal = -normal;
    }

    const auto eta = eta_i / eta_t;
    const auto k = 1.0f - eta * eta * (1.0f - cos_i * cos_i);
    return k < 0.0f ? glm::vec3(0.0f) : glm::normalize(eta * ray_in_dir + (eta * cos_i - glm::sqrt(k)) * correct_normal);
}
FUNC_QUALIFIER static inline float fresnel(const glm::vec3 &observation_dir, const glm::vec3 &normal, float ior) {
    auto cos_i = clamp(-1.0f, 1.0f, glm::dot(observation_dir, normal));
    auto eta_i = 1.0f;
    auto eta_t = ior;
    if (cos_i > 0.0f) {
        swap(eta_i, eta_t);
    }
    // Compute sin_t using Snell's law
    const auto sin_t = eta_i / eta_t * glm::sqrt(glm::max(0.0f, 1.0f - cos_i * cos_i));
    // Total internal reflection
    if (sin_t >= 1.0f) {
        return 1.0f;
    } else {
        const auto cos_t = glm::sqrt(glm::max(0.0f, 1.0f - sin_t * sin_t));
        cos_i = glm::abs(cos_i);
        const auto rs = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t));
        const auto rp = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t));
        return (rs * rs + rp * rp) / 2;
    }
}
namespace Microfacet {
    FUNC_QUALIFIER inline float distribution(float normal_dot_micro_surface_normal, float roughness_sq) {
        const auto normal_dot_micro_surface_normal_sq = normal_dot_micro_surface_normal * normal_dot_micro_surface_normal;
        auto denominator = normal_dot_micro_surface_normal_sq * (roughness_sq - 1.0f) + 1.0f;
        denominator = Pi * denominator * denominator;
        return roughness_sq / denominator;
    }

    FUNC_QUALIFIER inline glm::vec3 fresnel_schlick(float micro_surface_normal_dot_ray_out_dir, const glm::vec3 &f0) {
        return f0 + (glm::vec3(1.0f) - f0) * Math::pow5(1.0f - micro_surface_normal_dot_ray_out_dir);
    }

    FUNC_QUALIFIER inline float geometry(float normal_dot_light_source_dir, float normal_dot_observer_dir, float roughness) {
        return 2.0f / Math::lerp(glm::abs(2 * normal_dot_light_source_dir * normal_dot_observer_dir), glm::abs(normal_dot_light_source_dir + normal_dot_observer_dir), roughness);
    }

    // GGX NDF Sampling
    // https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html
    // https://agraphicsguynotes.com/posts/sample_microfacet_brdf/
    FUNC_QUALIFIER inline glm::vec3 sample_micro_surface(RNG& rng, const glm::vec3 &normal, float roughness_sq) {
        const auto r0 = rng.sample1D();
        const auto r1 = rng.sample1D();
        const auto theta = glm::acos(glm::sqrt((1.0f - r0) / ((roughness_sq - 1.0f) * r0 + 1.0f)));
        const auto phi = 2 * Pi * r1;

        const auto local_micro_surface_normal = Math::polar_to_cartesian(theta, phi);
        return Math::local_to_world(local_micro_surface_normal, normal);
    }

    FUNC_QUALIFIER inline float pdf_micro_surface(float normal_dot_micro_surface_normal, float roughness_sq) {
        // importance sampling on NDF
        const auto normal_dot_micro_surface_normal_abs = glm::abs(normal_dot_micro_surface_normal);
        return (distribution(normal_dot_micro_surface_normal_abs, roughness_sq) * normal_dot_micro_surface_normal_abs);
    }

    FUNC_QUALIFIER inline glm::vec3 outward_micro_surface_normal(const glm::vec3 &ray_source_dir, const glm::vec3 &ray_out_dir, bool is_same_side, bool is_surface_outward, float ior) {
        if (is_same_side) {
            // reflection
            if (is_surface_outward) {
                return glm::normalize(ray_source_dir + ray_out_dir);
            } else {
                return -glm::normalize(ray_source_dir + ray_out_dir);
            }
        } else {
            // refraction
            if (is_surface_outward) {
                return -glm::normalize(ray_out_dir + ray_source_dir * ior);
            } else {
                return -glm::normalize(ior * ray_out_dir + ray_source_dir);
            }
        }
    }

    FUNC_QUALIFIER inline float reflect_jacobian(float micro_surface_normal_dot_ray_out_dir) {
        return micro_surface_normal_dot_ray_out_dir == 0.0f ? 0.0f : 1.0f / (4 * glm::abs(micro_surface_normal_dot_ray_out_dir));
    }

    FUNC_QUALIFIER inline float refract_jacobian(float micro_surface_normal_dot_ray_source_dir, float ior_in, float micro_surface_normal_dot_ray_out_dir, float ior_out) {
        auto denominator = ior_in * micro_surface_normal_dot_ray_source_dir + ior_out * micro_surface_normal_dot_ray_out_dir;
        denominator *= denominator;
        return denominator == 0.0f ? 0.0f : (ior_out * ior_out * glm::abs(micro_surface_normal_dot_ray_out_dir)) / denominator;
    }

    FUNC_QUALIFIER static glm::vec3 GTR2Sample(glm::vec3 n, glm::vec3 wo, float alpha, glm::vec2 r) {
        glm::mat3 transMat = Math::localRefMatrix(n);
        glm::mat3 transInv = glm::inverse(transMat);

        glm::vec3 vh = glm::normalize((transInv * wo) * glm::vec3(alpha, alpha, 1.f));

        float lenSq = vh.x * vh.x + vh.y * vh.y;
        glm::vec3 t = lenSq > 0.f ? glm::vec3(-vh.y, vh.x, 0.f) / sqrt(lenSq) : glm::vec3(1.f, 0.f, 0.f);
        glm::vec3 b = glm::cross(vh, t);

        glm::vec2 p = Math::toConcentricDisk(r.x, r.y);
        float s = 0.5f * (vh.z + 1.f);
        p.y = (1.f - s) * glm::sqrt(1.f - p.x * p.x) + s * p.y;

        glm::vec3 h = t * p.x + b * p.y + vh * glm::sqrt(glm::max(0.f, 1.f - glm::dot(p, p)));
        h = glm::vec3(h.x * alpha, h.y * alpha, glm::max(0.f, h.z));
        return glm::normalize(transMat * h);
    }
} // namespace Microfacet

struct Material {
    enum Type {
        Lambertian,
        MetallicWorkflow,
        Glass,
    };

    int type = Type::Lambertian;
    glm::vec3 _emission = glm::vec3(0.); // for light source
    glm::vec3 _albedo = glm::vec3(.9f);  // for diffuse & metallic
    float _metallic = 0.f;               // for metallic
    float _roughness = 1.f;              // for metallic & frosted glass
    float _ior = 0.f;                    // for frosted glass

    FUNC_QUALIFIER bool emitting() {
        return glm::dot(_emission, _emission) > 0.0f;
    }
    FUNC_QUALIFIER glm::vec3 emission() {
        return _emission;
    }

    FUNC_QUALIFIER float alpha() {
        return glm::sqrt(_roughness);
    }

    FUNC_QUALIFIER bool effectivelySmooth() {
        return alpha() < 1e-3f;
    }

    FUNC_QUALIFIER bool hasSpecular() {
        return (type == Type::MetallicWorkflow && effectivelySmooth() ) || 
                type == Type::Glass;
    }

    FUNC_QUALIFIER void regularize() {
        auto alpha_x = alpha();
        if (alpha_x < 0.3f)
        {
            alpha_x = glm::clamp(2 * alpha_x, 0.1f, 0.3f);
            _roughness = alpha_x * alpha_x;
        }
    }

    FUNC_QUALIFIER glm::vec3 metallicWorkflowSample(glm::vec3 n, glm::vec3 wo, glm::vec3 r) {
        float alpha = _roughness * _roughness;

        if (r.z > (1.f / (2.f - _metallic))) {
            return Math::sampleHemisphereCosine(n, r.x, r.y);
        }
        else {
            glm::vec3 h = Microfacet::GTR2Sample(n, wo, alpha, glm::vec2(r));
            return -glm::reflect(wo, h);
        }
    }

    // Given the direction of the observer, calculate a random ray source direction.
    FUNC_QUALIFIER glm::vec3 sample(RNG& rng, const glm::vec3 &ray_out_dir, const glm::vec3 &normal) {
        switch (type) {
        case Type::Lambertian: {
            return Math::sampleHemisphereCosine(normal, rng.sample1D(), rng.sample1D());
        }
        case Type::MetallicWorkflow: {
            // another importance sampling method.
            // since Normal Distribution Function dominates the shape of the micro surface,
            // we first sample normal and generate correspond ray direction, which would contributes
            // more to the final radiance than a ray with random direction.
            const auto micro_surface_normal = Microfacet::sample_micro_surface(rng, normal, Math::square(_roughness));
            const auto observation_dir = -ray_out_dir;
            return reflect(observation_dir, micro_surface_normal); // trace back
        }
        case Type::Glass: {
            // randomly choose a micro surface
            const auto micro_surface_normal = Microfacet::sample_micro_surface(rng, normal, Math::square(_roughness));
            const auto observation_dir = -ray_out_dir;

            const auto f = fresnel(observation_dir, micro_surface_normal, _ior);

            // trace back
            if (rng.sample1D() < f) {
                // reflection
                return reflect(observation_dir, micro_surface_normal);
            } else {
                // refraction(transmission)
                return refract(observation_dir, micro_surface_normal, _ior);
            }
        }
        default:
            return glm::vec3(0.0f);
        }
    }

    FUNC_QUALIFIER static float schlickG(float cosTheta, float alpha) {
        float a = alpha * .5f;
        return cosTheta / (cosTheta * (1.f - a) + a);
    }

    FUNC_QUALIFIER inline float smithG(float cosWo, float cosWi, float alpha) {
        return schlickG(glm::abs(cosWo), alpha) * schlickG(glm::abs(cosWi), alpha);
    }

    FUNC_QUALIFIER static float GTR2Distrib(float cosTheta, float alpha) {
        if (cosTheta < 1e-6f) {
            return 0.f;
        }
        float aa = alpha * alpha;
        float nom = aa;
        float denom = cosTheta * cosTheta * (aa - 1.f) + 1.f;
        denom = denom * denom * Pi;
        return nom / denom;
    }

    FUNC_QUALIFIER static float GTR2Pdf(glm::vec3 n, glm::vec3 m, glm::vec3 wo, float alpha) {
        return GTR2Distrib(glm::dot(n, m), alpha) * schlickG(glm::dot(n, wo), alpha) *
            Math::absDot(m, wo) / Math::absDot(n, wo);
    }

    FUNC_QUALIFIER float metallicWorkflowPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        glm::vec3 h = glm::normalize(wo + wi);
        return glm::mix(
            Math::satDot(n, wi) * PiInv,
            GTR2Pdf(n, h, wo, _roughness * _roughness) / (4.f * Math::absDot(h, wo)),
            1.f / (2.f - _metallic)
        );
    }

    // Given the sampled ray source direction, the direction of ray out and a normal vector,
    // calculate its value of PDF (probability distribution function).
    FUNC_QUALIFIER float pdf(const glm::vec3 &ray_source_dir, const glm::vec3 &ray_out_dir, const glm::vec3 &normal) {
        switch (type) {
        case Type::Lambertian: {
            // uniformly sampling from hemisphere results in probability 1 / (2 * PI)
            return glm::dot(ray_out_dir, normal) > 0.0f ? 0.5f * PiInv : 0.0f;
        }
        case Type::MetallicWorkflow: {
            const auto check_ray_dir = glm::dot(normal, ray_source_dir) * glm::dot(normal, ray_out_dir);
            if (check_ray_dir <= 0.0f)
                return 0.0f; // no refraction

            const auto micro_surface_normal = glm::normalize(ray_source_dir + ray_out_dir);
            const auto normal_dot_micro_surface_normal = glm::dot(normal, micro_surface_normal);
            const auto micro_surface_normal_dot_ray_out_dir = glm::dot(micro_surface_normal, ray_out_dir);

            const auto pdf_micro_surface = Microfacet::pdf_micro_surface(normal_dot_micro_surface_normal, Math::square(_roughness));
            const auto jacobian = Microfacet::reflect_jacobian(micro_surface_normal_dot_ray_out_dir);

            return pdf_micro_surface * jacobian;
        }
        case Type::Glass: {
            const auto normal_dot_ray_source_dir = glm::dot(normal, ray_source_dir);
            const auto normal_dot_ray_out_dir = glm::dot(normal, ray_out_dir);
            const auto observation_dir = -ray_out_dir;

            const auto check_ray_dir = normal_dot_ray_source_dir * normal_dot_ray_out_dir;
            if (check_ray_dir == 0.0f)
                return 0.0f;

            const auto is_same_side = check_ray_dir > 0.0f;
            const auto is_surface_outward = normal_dot_ray_out_dir > 0.0f;

            const auto micro_surface_normal = Microfacet::outward_micro_surface_normal(ray_source_dir, ray_out_dir, is_same_side, is_surface_outward, _ior);
            const auto f = fresnel(observation_dir, micro_surface_normal, _ior);

            const auto normal_dot_micro_surface_normal = glm::dot(normal, micro_surface_normal);
            const auto pdf_micro_surface = Microfacet::pdf_micro_surface(normal_dot_micro_surface_normal, Math::square(_roughness));

            const auto micro_surface_normal_dot_ray_source_dir = glm::dot(micro_surface_normal, ray_source_dir);
            const auto micro_surface_normal_dot_ray_out_dir = glm::dot(micro_surface_normal, ray_out_dir);

            if (is_same_side) {
                // reflection
                const auto jacobian = Microfacet::reflect_jacobian(micro_surface_normal_dot_ray_out_dir);
                return pdf_micro_surface * f * jacobian;
            } else {
                // refraction
                const auto ior_in = normal_dot_ray_source_dir < 0.0f ? _ior : 1.0f;
                const auto ior_out = normal_dot_ray_out_dir < 0.0f ? _ior : 1.0f;
                const auto jacobian = Microfacet::refract_jacobian(micro_surface_normal_dot_ray_source_dir, ior_in, micro_surface_normal_dot_ray_out_dir, ior_out);
                return pdf_micro_surface * (1.0f - f) * jacobian;
            }
        }
        }
        return 0.0f;
    }
    // Given the directions of the ray source and ray out and a normal vector,
    // calculate its contribution from BSDF (bidirectional scattering distribution function).
    FUNC_QUALIFIER glm::vec3 bsdf(const glm::vec3 &ray_source_dir, const glm::vec3 &ray_out_dir, const glm::vec3 &normal) {
        switch (type) {
        case Type::Lambertian: {
            return glm::dot(ray_out_dir, normal) > 0.0f ? _albedo * PiInv : glm::vec3(0.0f);
        }
        case Type::MetallicWorkflow: {
            // GGX BRDF
            const auto check_ray_dir = glm::dot(normal, ray_source_dir) * glm::dot(normal, ray_out_dir);
            if (check_ray_dir <= 0.0f)
                return glm::vec3(0.0f); // no refraction

            const auto micro_surface_normal = glm::normalize(ray_source_dir + ray_out_dir);
            const auto normal_dot_micro_surface_normal = glm::dot(normal, micro_surface_normal);
            const auto micro_surface_normal_dot_ray_source_dir = glm::dot(micro_surface_normal, ray_source_dir);
            const auto micro_surface_normal_dot_ray_out_dir = glm::dot(micro_surface_normal, ray_out_dir);

            const auto D = Microfacet::distribution(normal_dot_micro_surface_normal, Math::square(_roughness));
            const auto G = Microfacet::geometry(glm::dot(normal, ray_source_dir), glm::dot(normal, ray_out_dir), _roughness);

            // Metal-roughness workflow
            const glm::vec3 f0_base(0.04f);
            const auto f0 = Math::lerp(f0_base, _albedo, _metallic);
            const auto F = Microfacet::fresnel_schlick(micro_surface_normal_dot_ray_out_dir, f0);
            const auto kd = (glm::vec3(1.0f) - F) * (1.0f - _metallic);

            // Lambert diffuse
            const auto diffuse = kd * _albedo * PiInv;
            // Cook–Torrance Specular (original denominator is merged into G for Smith-Joint approximation)
            const auto specular = D * F * G / 4.0f;

            // BSDF
            return diffuse + specular;
        }
        case Type::Glass: {
            const auto normal_dot_ray_source_dir = glm::dot(normal, ray_source_dir);
            const auto normal_dot_ray_out_dir = glm::dot(normal, ray_out_dir);
            const auto check_ray_dir = normal_dot_ray_source_dir * normal_dot_ray_out_dir;
            if (check_ray_dir == 0.0f)
                return glm::vec3(0.0f);

            const auto observation_dir = -ray_out_dir;
            const auto is_same_side = check_ray_dir > 0.0f;
            const auto is_surface_outward = normal_dot_ray_out_dir > 0.0f;

            const auto micro_surface_normal = Microfacet::outward_micro_surface_normal(ray_source_dir, ray_out_dir, is_same_side, is_surface_outward, _ior);

            const auto normal_dot_micro_surface_normal = glm::dot(normal, micro_surface_normal);
            const auto micro_surface_normal_dot_ray_source_dir = glm::dot(micro_surface_normal, ray_source_dir);
            const auto micro_surface_normal_dot_ray_out_dir = glm::dot(micro_surface_normal, ray_out_dir);

            const auto D = Microfacet::distribution(normal_dot_micro_surface_normal, Math::square(_roughness));
            const auto G = Microfacet::geometry(micro_surface_normal_dot_ray_source_dir, micro_surface_normal_dot_ray_out_dir, _roughness);
            const auto F = fresnel(observation_dir, micro_surface_normal, _ior);

            if (is_same_side) {
                // reflection
                return glm::vec3(D * F * G / 4.0f); // Cook–Torrance Specular (original denominator is merged into G for Smith-Joint approximation)
            } else {
                // refraction
                const auto ior_in = normal_dot_ray_source_dir < 0.0f ? _ior : 1.0f;
                const auto ior_out = normal_dot_ray_out_dir < 0.0f ? _ior : 1.0f;

                // Transmission (original denominator is merged into G for Smith-Joint approximation)
                float r = Microfacet::refract_jacobian(micro_surface_normal_dot_ray_source_dir, ior_in, micro_surface_normal_dot_ray_out_dir, ior_out) * abs(micro_surface_normal_dot_ray_source_dir) * D * (1.0f - F) * G;
                return glm::vec3(r);
            }
        }
        }
        return glm::vec3(0.0f);
    }

    CUDA_PORTABLE(Material);
};