#pragma once

#include "MathUtils.hpp"

FUNC_QUALIFIER inline glm::vec3 reflect(const glm::vec3 &ray_in_dir, const glm::vec3 &normal) {
    return ray_in_dir - glm::dot(ray_in_dir, normal) * normal * 2.f;
}
// If the ray is outside, make cos_i positive.
// If the ray is inside, invert the refractive indices and negate the normal.
FUNC_QUALIFIER inline glm::vec3 refract(const glm::vec3 &ray_in_dir, const glm::vec3 &normal, float ior) {
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
FUNC_QUALIFIER inline float fresnel(const glm::vec3 &observation_dir, const glm::vec3 &normal, float ior) {
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

    FUNC_QUALIFIER inline glm::vec3 sample_micro_surface(const glm::vec3 &r, const glm::vec3 &normal, float roughness_sq) {
        const auto r0 = r.x;
        const auto r1 = r.y;
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

    // Given the direction of the observer, calculate a random ray source direction.
    FUNC_QUALIFIER glm::vec3 sample(const glm::vec3 &r, const glm::vec3 &ray_out_dir, const glm::vec3 &normal) {
        switch (type) {
        case Type::Lambertian: {
            // uniformly sample the hemisphere
            const auto x1 = r.x;
            const auto x2 = r.y;
            const auto z = glm::abs(1.0f - 2.0f * x1);
            const auto r = glm::sqrt(1.0f - z * z);
            const auto phi = 2 * Pi * x2;

            // get local direction of the ray out
            const glm::vec3 local_ray_out_dir(r * glm::cos(phi), r * glm::sin(phi), z);

            // transform to the world space
            return Math::local_to_world(local_ray_out_dir, normal);
        }
        case Type::MetallicWorkflow: {
            const auto micro_surface_normal = Microfacet::sample_micro_surface(r, normal, Math::square(_roughness));
            const auto observation_dir = -ray_out_dir;
            return reflect(observation_dir, micro_surface_normal); // trace back
        }
        case Type::Glass: {
            // randomly choose a micro surface
            const auto micro_surface_normal = Microfacet::sample_micro_surface(r, normal, Math::square(_roughness));
            const auto observation_dir = -ray_out_dir;

            const auto f = fresnel(observation_dir, micro_surface_normal, _ior);

            // trace back
            if (r.z < f) {
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
    FUNC_QUALIFIER glm::vec3 fr(const glm::vec3 &ray_source_dir, const glm::vec3 &ray_out_dir, const glm::vec3 &normal) {
        switch (type) {
        case Type::Lambertian: {
            return glm::dot(ray_out_dir, normal) > 0.0f ? _albedo * PiInv : glm::vec3(0.0f);
        }
        case Type::MetallicWorkflow: {
            const auto check_ray_dir = glm::dot(normal, ray_source_dir) * glm::dot(normal, ray_out_dir);
            if (check_ray_dir <= 0.0f)
                return glm::vec3(0.0f); // no refraction

            const auto micro_surface_normal = glm::normalize(ray_source_dir + ray_out_dir);
            const auto normal_dot_micro_surface_normal = glm::dot(normal, micro_surface_normal);
            const auto micro_surface_normal_dot_ray_source_dir = glm::dot(micro_surface_normal, ray_source_dir);
            const auto micro_surface_normal_dot_ray_out_dir = glm::dot(micro_surface_normal, ray_out_dir);

            const auto D = Microfacet::distribution(normal_dot_micro_surface_normal, Math::square(_roughness));
            const auto G = Microfacet::geometry(micro_surface_normal_dot_ray_source_dir, micro_surface_normal_dot_ray_out_dir, _roughness);

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