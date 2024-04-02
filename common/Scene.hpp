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
#include "Vector.hpp"
#include <vector>
#include <string>
#include "CudaPortable.hpp"

class Scene {
public:
    // setting up options
    std::string name = "default";
    // todo: consider using a camera class to to store these options
    Vector3f camPos = Vector3f(0, 0, 0);
    int width = 1280;
    int height = 960;
    double fov = 40;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
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
    FUNC_QUALIFIER Intersection intersect(const Ray& ray) const;
    void buildBVH();
    FUNC_QUALIFIER Vector3f castRay(const Ray& eyeRay) const;
    FUNC_QUALIFIER void sampleLight(Intersection& pos, float& pdf) const;
    // FUNC_QUALIFIER bool trace(const Ray& ray, const std::vector<Object*>& objects, float& tNear, uint32_t& index, Object** hitObject);
    // std::tuple<Vector3f, Vector3f> HandleAreaLight(const AreaLight &light, const Vector3f &hitPoint, const Vector3f &N,
    //                                                const Vector3f &shadowPointOrig,
    //                                                const std::vector<Object *> &objects, uint32_t &index,
    //                                                const Vector3f &dir, float specularExponent);

    // creating the scene (adding objects and lights)

    // std::vector<std::unique_ptr<Light>> lights;

    // Compute reflection direction
    FUNC_QUALIFIER Vector3f reflect(const Vector3f& I, const Vector3f& N) const {
        return I - 2 * dotProduct(I, N) * N;
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
Vector3f Scene::castRay(const Ray& eyeRay) const {
    glm::vec3 accRadiance(0);

    Ray ray = eyeRay;
    Intersection intersec = Scene::intersect(ray);

    // if not hit anything, return background color
    if (!intersec.happened)
        return fromGlm(accRadiance);
    // if hit light, return light emission
    if (intersec.m->hasEmission())
        return intersec.m->getEmission();

    glm::vec3 throughput(1.0f);
    glm::vec3 wo = -ray.direction.toGlm();
    glm::vec3 normal = intersec.normal.toGlm();
    Material* material = intersec.m;
    auto mateiralType = material->getType();
    for (int depth = 0; depth < this->maxDepth; ++depth)
    {
        bool deltaBSDF = (mateiralType == Dielectric);

        if (mateiralType != Dielectric && glm::dot(normal, wo) < 0.f) {
            normal = -normal;
            intersec.normal = fromGlm(normal);
        }

        if (!deltaBSDF) {
            glm::vec3 radiance = intersec.emit.toGlm();
            glm::vec3 wi;
            float lightPdf;
            sampleLight(intersec, lightPdf);

            if (lightPdf > 0) {
                float BSDFPdf = material->pdf(fromGlm(wi), fromGlm(wo), fromGlm(normal));
                accRadiance += throughput * material->eval(fromGlm(wi), fromGlm(wo), fromGlm(normal)).toGlm() *
                    radiance * Math::satDot(normal, wi) / lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
            }
        }

		// BSDFSample sample;
		// material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);
        glm::vec3 dummy(0.0f); // todo: ???
        glm::vec3 sampleDir = material->sample(fromGlm(dummy), fromGlm(normal)).normalized().toGlm();
        glm::vec3 sampleBsdf = material->eval(ray.direction, fromGlm(sampleDir), fromGlm(normal)).toGlm();
        float samplePdf = material->pdf(ray.direction, fromGlm(sampleDir), fromGlm(normal));
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

        ray = Ray(intersec.coords, fromGlm(sampleDir), kEpsilon);

        glm::vec3 curPos = intersec.coords.toGlm();
        intersec = Scene::intersect(ray);
        if (!intersec.happened) {
            break;
        }

        wo = -ray.direction.toGlm();
        normal = intersec.normal.toGlm();
        material = intersec.m;
        mateiralType = material->getType();

        // hit light
        if (material->hasEmission()) {
            glm::vec3 radiance = material->getEmission().toGlm();

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

    return fromGlm(accRadiance);
}