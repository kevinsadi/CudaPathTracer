//
// Created by Göksu Güvendiren on 2019-05-14.
//

#pragma once

#include "BVH.hpp"
#include "CudaPortable.hpp"
// #include "Object.hpp"
#include "Ray.hpp"
#include "Settings.hpp"
#include "Triangle.hpp"
#include <string>
#include <unordered_map>
#include <vector>

enum struct Culling { NONE, BACK, FRONT };


class Scene {
public:
    // setting up options
    std::string name = "default";
    // todo: consider using a camera class to to store these options
    glm::vec3 camPos = glm::vec3(0, 0, 0);
    int width = 1280;
    int height = 960;
    double fov = 40;
    glm::vec3 backgroundColor = glm::vec3(0.235294, 0.67451, 0.843137);
    int maxDepth = 1;
    float RussianRoulette = 0.8;
    BVHAccel* bvh = nullptr;
    // -----------Editor Only-----------
    // [!] as polymorphic is not supported in CUDA, currently we only allow MeshTriangle
    std::vector<MeshTriangle*> meshes;
    std::unordered_map<MeshTriangle*, BVHAccel*> meshBvhMap;
    // ---------------------------------
    MeshTriangle** meshes_data = nullptr;
    BVHAccel** mesh_bvhs = nullptr;
    int num_meshes = 0;
    float sumLightArea = 0.0f;
    float sumLightPower = 0.0f;
    int num_lights = 0;

    Scene(int w, int h) : width(w), height(h) {}

    void Add(MeshTriangle* mesh) {
        meshes.push_back((MeshTriangle*)mesh);
        num_meshes = meshes.size();
        meshes_data = meshes.data();
        if (mesh->material.emitting()) {
            sumLightArea += mesh->area;
            sumLightPower += mesh->area * Math::luminance(mesh->material._emission) * 2.f * glm::pi<float>();
            ++num_lights;
        }
    }

    // const std::vector<Object *> &get_objects() const { return objects; }
    // const std::vector<std::unique_ptr<Light>> &get_lights() const { return
    // lights; }
    FUNC_QUALIFIER inline Intersection intersect(const Ray& ray) const;
    void buildBVH();
    // FUNC_QUALIFIER void TracePath(PathSegment& path);
    FUNC_QUALIFIER inline glm::vec3 castRay(RNG& rng, const Ray& eyeRay) const;
    FUNC_QUALIFIER inline void sampleLight(RNG& rng, Intersection& pos, float& pdf) const;
    // FUNC_QUALIFIER inline bool trace(const Ray& ray, const
    // std::vector<Object*>& objects, float& tNear, uint32_t& index, Object**
    // hitObject); std::tuple<glm::vec3, glm::vec3> HandleAreaLight(const
    // AreaLight &light, const glm::vec3 &hitPoint, const glm::vec3 &N,
    //                                                const glm::vec3
    //                                                &shadowPointOrig, const
    //                                                std::vector<Object *>
    //                                                &objects, uint32_t &index,
    //                                                const glm::vec3 &dir, float
    //                                                specularExponent);

    // creating the scene (adding objects and lights)

    // std::vector<std::unique_ptr<Light>> lights;

    // Compute reflection direction
    FUNC_QUALIFIER inline glm::vec3 reflect(const glm::vec3& I,
        const glm::vec3& N) const {
        return I - 2 * glm::dot(I, N) * N;
    }

    enum BuiltinScene {
        CornellBox,
    };
    static Scene CreateBuiltinScene(BuiltinScene sceneId, int maxDepth);

    CUDA_PORTABLE(Scene);
};

Intersection Scene::intersect(const Ray& ray) const {
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(RNG& rng, Intersection& pos, float& pdf) const {
    float emit_area_sum = sumLightArea;

    float p = rng.sample1D() * emit_area_sum;
    emit_area_sum = 0;
    // for (uint32_t k = 0; k < objects.size(); ++k) {
    //     auto object = objects[k];
    for (uint32_t k = 0; k < num_meshes; ++k) {
        auto mesh = meshes_data[k];
        auto meshBvh = mesh_bvhs[k];
        if (mesh->material.emitting()) {
            emit_area_sum += mesh->area;
            if (p <= emit_area_sum) {
                meshBvh->Sample(rng, pos, pdf);
                // pdf = 1.0f / meshBvh->root->area;
                // pdf /= num_lights;
                pos.emit = mesh->material.emission();
                break;
            }
        }
    }
}

// void Scene::TracePath(PathSegment& path)
// {
//     // 1. Monte Carlo Path Tracing: integration from all directions can be estimated 
//     // by averaging over radiance/pdf on each direction, I = 1/N * Σ(Li*bsdf*cos_theta/pdf)
//     // 2. since we are using iteration instead of recursion, we use a throughput to store the
//     // weight(bsdf*cos_theta/pdf) for an indirect ray
//     // reference: https://sites.cs.ucsb.edu/~lingqi/teaching/resources/GAMES101_Lecture_16.pdf, p43
//     glm::vec3& accRadiance = path.radiance;
//     glm::vec3& throughput = path.throughput;

//     Ray& ray = path.ray;
//     float& bsdfSamplePdf = path.bsdfSamplePdf;
//     const bool pathRegularization = false;
//     // const bool pathRegularization = true;
//     const bool enableRR = false;
//     // const bool enableRR = true;
//     bool& specularBounce = path.specularBounce;
//     bool& anyNonSpecularBounces = path.anyNonSpecularBounces;
    
//     if (path.remainingBounces == 0)
//     {
//         return;
//     }
//     Intersection& intersec = Scene::intersect(ray);
//     Material& material = intersec.m;

//     if (!intersec.happened)
//     {
//         // sample envmap
//         path.remainingBounces = 0;
//         return;
//     }

//     // add emission from surface hit by ray
//     if (material.emitting())
//     {
//         // disable MIS if the previous bounce was a specular bounce
//         if (depth == 0 || specularBounce)
//         {
//             accRadiance += throughput * material.emission();
//         }
//         else
//         {
//             glm::vec3 radiance = material.emission();

//             // float lightPdf = Math::pdfAreaToSolidAngle(
//             //     Math::luminance(radiance) * 2.f * glm::pi<float>() * intersec.triangleArea / sumLightPower,
//             //     ray.origin, intersec.coords, intersec.normal
//             // );
//             auto lightPdf = Math::luminance(radiance) * 2.f * glm::pi<float>() * intersec.triangleArea / sumLightPower;
            
//             float weight = Math::powerHeuristic(bsdfSamplePdf, lightPdf);
//             accRadiance += throughput * radiance * weight;
//         }
//     }

//     if (pathRegularization && anyNonSpecularBounces)
//     {
//         material.regularize();
//     }

//     // direct lighting
//     if (!specularBounce) {
//         Intersection lightSample;
//         float lightSamplePdf = 0.0f;
//         sampleLight(path.rng, lightSample, lightSamplePdf);
//         if (lightSamplePdf > 0.0f)
//         {
//             glm::vec3 p = intersec.coords;
//             glm::vec3 x = lightSample.coords;
//             glm::vec3 wo = -ray.direction;
//             glm::vec3 wi = glm::normalize(x - p);
//             Ray shadowRay(p + wi * Epsilon5, wi);
//             Intersection shadowIntersec = Scene::intersect(shadowRay);
//             if (shadowIntersec.distance - glm::length(x - p) > -Epsilon4) {
//                 glm::vec3 radiance = lightSample.emit;
//                 glm::vec3 bsdf = material.bsdf(wi, wo, intersec.normal);
//                 float cos_theta = Math::satDot(intersec.normal, wi);
//                 float cos_theta_prime = Math::satDot(lightSample.normal, -wi);
//                 float r2 = glm::dot(x - p, x - p);
//                 float bsdfPdf = material.pdf(wi, wo, intersec.normal);
//                 // float weight = Math::powerHeuristic(lightSamplePdf, bsdfPdf);
//                 float lightPdf = Math::luminance(radiance) * 2.f * glm::pi<float>() * lightSample.triangleArea / sumLightPower;
//                 float weight = Math::powerHeuristic(lightPdf, bsdfPdf);
//                 // float lightPdf = Math::pdfAreaToSolidAngle(
//                 //     Math::luminance(radiance) * 2.f * glm::pi<float>() * lightSample.triangleArea / sumLightPower,
//                 //     p, x, lightSample.normal
//                 // );
//                 // float weight = Math::powerHeuristic(lightPdf, bsdfSamplePdf);
//                 // float weight = Math::powerHeuristic(lightPdf, bsdfSamplePdf);
//                 // float weight = 1.0f;
//                 // accRadiance += throughput * emit * bsdf * cos_theta * cos_theta_prime / r2 / lightSamplePdf;
//                 accRadiance += throughput * radiance * bsdf * cos_theta * cos_theta_prime / r2 / lightSamplePdf * weight;
//             }
//         }
//     }

//     // sample bsdf to get new path direction
//     // bool deltaSample = !specularBounce; // not consider transmission yet
//     bool deltaSample = false; // not consider transmission yet
//     glm::vec3 wo = -ray.direction;
//     glm::vec3 wi = material.sample(path.rng, wo, intersec.normal);
//     glm::vec3 p = intersec.coords;
//     glm::vec3 bsdf = material.bsdf(wi, wo, intersec.normal);
//     bsdfSamplePdf = material.pdf(wi, wo, intersec.normal);
//     if (bsdfSamplePdf < Epsilon5) {
//         break;
//     }

//     // update path state
//     specularBounce = material.hasSpecular();
//     anyNonSpecularBounces |= !specularBounce;
//     float cos_theta = Math::absDot(intersec.normal, wi);
//     throughput *= bsdf
//         * (deltaSample ? 1.f : cos_theta)
//         / bsdfSamplePdf;
//     // generate new ray
//     ray = Ray(p + wi * Epsilon5, wi);

//     // Russian Roulette
//     if (enableRR)
//     {
//         auto rrProb = path.rng.sample1D();
//         if (depth > 1) {
//             float q = 1 - RussianRoulette;
//             if (rrProb < q)
//             {
//                 break;
//             }
//             throughput /= (1.f - q);
//         }
//     }
//     if (isnan(accRadiance.x) || isnan(accRadiance.y) || isnan(accRadiance.z) || isinf(accRadiance.x) || isinf(accRadiance.y) || isinf(accRadiance.z)) {
//         accRadiance = glm::vec3(0.0f);
//     }
// }

glm::vec3 Scene::castRay(RNG& rng, const Ray& eyeRay) const {
    // 1. Monte Carlo Path Tracing: integration from all directions can be estimated 
    // by averaging over radiance/pdf on each direction, I = 1/N * Σ(Li*bsdf*cos_theta/pdf)
    // 2. since we are using iteration instead of recursion, we use a throughput to store the
    // weight(bsdf*cos_theta/pdf) for an indirect ray
    // reference: https://sites.cs.ucsb.edu/~lingqi/teaching/resources/GAMES101_Lecture_16.pdf, p43

    // path states
    glm::vec3 accRadiance(0.f), throughput(1.f);
    Ray ray = eyeRay;
    float bsdfSamplePdf = 0.0f;
    bool specularBounce = false, anyNonSpecularBounces = false;

    // configurations
    const bool pathRegularization = false;
    const bool enableRR = false;
    const bool sampleDirectLighting = true;
    const bool sampleBsdfLighting = true;

    for (int depth = 0; depth < this->maxDepth; depth++) {
        Intersection& intersec = Scene::intersect(ray);
        Material& material = intersec.m;

        if (!intersec.happened)
        {
            // sample envmap
            break;
        }

        // add emission from surface hit by ray
        if (material.emitting() && sampleBsdfLighting)
        {
            // disable MIS if the previous bounce was a specular bounce
            if (depth == 0 || specularBounce)
            {
                accRadiance += throughput * material.emission();
            }
            else
            {
                glm::vec3 radiance = material.emission();

                // float lightPdf = Math::pdfAreaToSolidAngle(
                //     Math::luminance(radiance) * 2.f * glm::pi<float>() * intersec.triangleArea / sumLightPower,
                //     ray.origin, intersec.coords, intersec.normal
                // );
                auto lightPdf = Math::luminance(radiance) * 2.f * glm::pi<float>() * intersec.triangleArea / sumLightPower;
                
                float weight = Math::powerHeuristic(bsdfSamplePdf, lightPdf);
                accRadiance += throughput * radiance * weight;
            }
        }

        if (pathRegularization && anyNonSpecularBounces)
        {
            material.regularize();
        }

        // direct lighting
        if (!specularBounce && sampleDirectLighting) {
            Intersection lightSample;
            float lightSamplePdf = 0.0f;
            sampleLight(rng, lightSample, lightSamplePdf);
            if (lightSamplePdf > 0.0f)
            {
                glm::vec3 p = intersec.coords;
                glm::vec3 x = lightSample.coords;
                glm::vec3 wo = -ray.direction;
                glm::vec3 wi = glm::normalize(x - p);
                Ray shadowRay(p + wi * Epsilon5, wi);
                Intersection shadowIntersec = Scene::intersect(shadowRay);
                if (shadowIntersec.distance - glm::length(x - p) > -Epsilon4) {
                    glm::vec3 radiance = lightSample.emit;
                    glm::vec3 bsdf = material.bsdf(wi, wo, intersec.normal);
                    float cos_theta = Math::satDot(intersec.normal, wi);
                    float cos_theta_prime = Math::satDot(lightSample.normal, -wi);
                    float r2 = glm::dot(x - p, x - p);
                    float bsdfPdf = material.pdf(wi, wo, intersec.normal);
                    // float weight = Math::powerHeuristic(lightSamplePdf, bsdfPdf);
                    float lightPdf = Math::luminance(radiance) * 2.f * glm::pi<float>() * lightSample.triangleArea / sumLightPower;
                    float weight = Math::powerHeuristic(lightPdf, bsdfPdf);
                    // float lightPdf = Math::pdfAreaToSolidAngle(
                    //     Math::luminance(radiance) * 2.f * glm::pi<float>() * lightSample.triangleArea / sumLightPower,
                    //     p, x, lightSample.normal
                    // );
                    // float weight = Math::powerHeuristic(lightPdf, bsdfSamplePdf);
                    // float weight = Math::powerHeuristic(lightPdf, bsdfSamplePdf);
                    // float weight = 1.0f;
                    // accRadiance += throughput * emit * bsdf * cos_theta * cos_theta_prime / r2 / lightSamplePdf;
                    accRadiance += throughput * radiance * bsdf * cos_theta * cos_theta_prime / r2 / lightSamplePdf * weight;
                }
            }
        }

        // sample bsdf to get new path direction
        // bool deltaSample = !specularBounce; // not consider transmission yet
        bool deltaSample = false; // not consider transmission yet
        glm::vec3 wo = -ray.direction;
        glm::vec3 wi = material.sample(rng, wo, intersec.normal);
        glm::vec3 p = intersec.coords;
        glm::vec3 bsdf = material.bsdf(wi, wo, intersec.normal);
        bsdfSamplePdf = material.pdf(wi, wo, intersec.normal);
        if (bsdfSamplePdf < Epsilon5) {
            break;
        }

        // update path state
        specularBounce = material.hasSpecular();
        anyNonSpecularBounces |= !specularBounce;
        float cos_theta = Math::absDot(intersec.normal, wi);
        throughput *= bsdf
            * (deltaSample ? 1.f : cos_theta)
            / bsdfSamplePdf;

        // generate new ray
        ray = Ray(p + wi * Epsilon5, wi);

        // Russian Roulette
        if (enableRR)
        {
            auto rrProb = rng.sample1D();
            if (depth > 1) {
                float q = 1 - RussianRoulette;
                if (rrProb < q)
                {
                    break;
                }
                throughput /= (1.f - q);
            }
        }
    }
    if (isnan(accRadiance.x) || isnan(accRadiance.y) || isnan(accRadiance.z) || isinf(accRadiance.x) || isinf(accRadiance.y) || isinf(accRadiance.z)) {
        return glm::vec3(0.0f);
    }
    return accRadiance;
}