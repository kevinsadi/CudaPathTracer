//
// Created by Göksu Güvendiren on 2019-05-14.
//

#pragma once

#include "AreaLight.hpp"
#include "BVH.hpp"
#include "Light.hpp"
#include "Object.hpp"
#include "Triangle.hpp"
#include "Ray.hpp"
#include <vector>
#include <string>
#include "CudaPortable.hpp"

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
    // [!] as polymorphic is not supported in CUDA, currently we only allow MeshTriangle
    // std::vector<Object *> objects;
    std::vector<MeshTriangle*> meshes; // [!] for build scene stage only, dont use it in rendering
    MeshTriangle** meshes_data = nullptr;
    int num_meshes = 0;
    std::vector<Material*> materials;
    Material** materials_data = nullptr;
    int num_materials = 0;

    Scene(int w, int h) : width(w), height(h) {}

    void Add(Object* object) {
        if (typeid(*object) != typeid(MeshTriangle))
        {
            throw std::runtime_error("only support MeshTriangle");
        }
        // objects.push_back(object); 
        meshes.push_back((MeshTriangle*)object);
        num_meshes = meshes.size();
        meshes_data = meshes.data();
    }
    void AddMaterial(Material* material) {
        materials.push_back(material);
        materials_data = materials.data();
        num_materials = materials.size();
    }
    // void Add(std::unique_ptr<Light> light) { lights.push_back(std::move(light)); }

    // const std::vector<Object *> &get_objects() const { return objects; }
    // const std::vector<std::unique_ptr<Light>> &get_lights() const { return lights; }
    FUNC_QUALIFIER inline Intersection intersect(const Ray& ray) const;
    void buildBVH();
    FUNC_QUALIFIER inline glm::vec3 castRay(const Ray& eyeRay) const;
    FUNC_QUALIFIER inline void sampleLight(Intersection& pos, float& pdf) const;
    // FUNC_QUALIFIER inline bool trace(const Ray& ray, const std::vector<Object*>& objects, float& tNear, uint32_t& index, Object** hitObject);
    // std::tuple<glm::vec3, glm::vec3> HandleAreaLight(const AreaLight &light, const glm::vec3 &hitPoint, const glm::vec3 &N,
    //                                                const glm::vec3 &shadowPointOrig,
    //                                                const std::vector<Object *> &objects, uint32_t &index,
    //                                                const glm::vec3 &dir, float specularExponent);

    // creating the scene (adding objects and lights)

    // std::vector<std::unique_ptr<Light>> lights;

    // Compute reflection direction
    FUNC_QUALIFIER inline glm::vec3 reflect(const glm::vec3& I, const glm::vec3& N) const {
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

void Scene::sampleLight(Intersection& pos, float& pdf) const {
    float emit_area_sum = 0;
    // for (uint32_t k = 0; k < objects.size(); ++k) {
    //     auto object = objects[k];
    for (uint32_t k = 0; k < num_meshes; ++k) {
        auto object = meshes_data[k];
        if (object->hasEmit()) {
            emit_area_sum += object->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    // for (uint32_t k = 0; k < objects.size(); ++k) {
    //     auto object = objects[k];
    for (uint32_t k = 0; k < num_meshes; ++k) {
        auto object = meshes_data[k];
        if (object->hasEmit()) {
            emit_area_sum += object->getArea();
            if (p <= emit_area_sum) {
                object->Sample(pos, pdf);
                break;
            }
        }
    }
}

#define kEpsilon 0.0005

// Implementation of Path Tracing
glm::vec3 Scene::castRay(const Ray& eyeRay) const {
    glm::vec3 accRadiance(0);

    Ray ray = eyeRay;
    Intersection intersec = Scene::intersect(ray);

    // if not hit anything, return background color
    if (!intersec.happened)
        return accRadiance;
    // if hit light, return light emission
    if (intersec.m->hasEmission())
        return intersec.m->getEmission();

    glm::vec3 throughput(1.0f);
    glm::vec3 wo = -ray.direction;
    glm::vec3 normal = intersec.normal;
    Material* material = intersec.m;
    auto mateiralType = material->getType();
    for (int depth = 0; depth < this->maxDepth; ++depth)
    {
        bool deltaBSDF = (mateiralType == Dielectric);

        if (mateiralType != Dielectric && glm::dot(normal, wo) < 0.f) {
            normal = -normal;
            intersec.normal = normal;
        }

        if (!deltaBSDF) {
            glm::vec3 radiance = intersec.emit;
            glm::vec3 wi;
            float lightPdf;
            sampleLight(intersec, lightPdf);

            if (lightPdf > 0) {
                float BSDFPdf = material->pdf(wi, wo, normal);
                accRadiance += throughput * material->eval(wi, wo, normal) *
                    radiance * Math::satDot(normal, wi) / lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
            }
        }

        // BSDFSample sample;
        // material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);
        glm::vec3 dummy(0.0f); // todo: ???
        glm::vec3 sampleDir = glm::normalize(material->sample(dummy, normal));
        glm::vec3 sampleBsdf = material->eval(ray.direction, sampleDir, normal);
        float samplePdf = material->pdf(ray.direction, sampleDir, normal);
        // if (sample.type == BSDFSampleType::Invalid) {
        //     // Terminate path if sampling fails
        //     break;
        // }
        if (samplePdf < 1e-8f) {
            break;
        }

        // bool deltaSample = (sample.type & Specular);
        bool deltaSample = false;
        throughput *= sampleBsdf / samplePdf *
            (deltaSample ? 1.f : Math::absDot(normal, sampleDir));

        ray = Ray(intersec.coords, sampleDir, kEpsilon);

        glm::vec3 curPos = intersec.coords;
        intersec = Scene::intersect(ray);
        if (!intersec.happened) {
            break;
        }

        wo = -ray.direction;
        normal = intersec.normal;
        material = intersec.m;
        mateiralType = material->getType();

        // hit light
        if (material->hasEmission()) {
            glm::vec3 radiance = material->getEmission();

            // float lightPdf = Math::pdfAreaToSolidAngle(
            //     Math::luminance(radiance) * 2.f * glm::pi<float>() * scene->getPrimitiveArea(intersec.primId) * scene->sumLightPowerInv,
            //     curPos, intersec.pos, intersec.norm
            // );
            float lightPdf;
            sampleLight(intersec, lightPdf);

            float weight = deltaSample ? 1.f : Math::powerHeuristic(samplePdf, lightPdf);
            accRadiance += radiance * throughput * weight;
            break;
        }
    }

    return accRadiance;
}