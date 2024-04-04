//
// Created by Göksu Güvendiren on 2019-05-14.
//

#pragma once

#include "BVH.hpp"
#include "CudaPortable.hpp"
#include "Object.hpp"
#include "Ray.hpp"
#include "Settings.hpp"
#include "Triangle.hpp"
#include <string>
#include <vector>

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
    BVHAccel *bvh = nullptr;
    // [!] as polymorphic is not supported in CUDA, currently we only allow
    // MeshTriangle std::vector<Object *> objects;
    std::vector<MeshTriangle *> meshes; // [!] for build scene stage only, dont use it in rendering
    MeshTriangle **meshes_data = nullptr;
    int num_meshes = 0;
    std::vector<Material> materials;
    Material *materials_data = nullptr;
    int num_materials = 0;

    Scene(int w, int h) : width(w), height(h) {}

    void Add(Object *object) {
        if (typeid(*object) != typeid(MeshTriangle)) {
            throw std::runtime_error("only support MeshTriangle");
        }
        // objects.push_back(object);
        meshes.push_back((MeshTriangle *)object);
        num_meshes = meshes.size();
        meshes_data = meshes.data();
    }

    void AddMaterial(Material material) {
        materials.push_back(material);
        materials_data = materials.data();
        num_materials = materials.size();
    }

    // const std::vector<Object *> &get_objects() const { return objects; }
    // const std::vector<std::unique_ptr<Light>> &get_lights() const { return
    // lights; }
    FUNC_QUALIFIER inline Intersection intersect(const Ray &ray) const;
    void buildBVH();
    FUNC_QUALIFIER inline glm::vec3 castRay(const Ray &eyeRay) const;
    FUNC_QUALIFIER inline void sampleLight(Intersection &pos, float &pdf) const;
    FUNC_QUALIFIER inline float lightAreaSum() const;
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
    FUNC_QUALIFIER inline glm::vec3 reflect(const glm::vec3 &I,
                                            const glm::vec3 &N) const {
        return I - 2 * glm::dot(I, N) * N;
    }

    enum BuiltinScene {
        CornellBox,
    };
    static Scene CreateBuiltinScene(BuiltinScene sceneId, int maxDepth);

    CUDA_PORTABLE(Scene);
};

Intersection Scene::intersect(const Ray &ray) const {
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const {
    float emit_area_sum = lightAreaSum();
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    // for (uint32_t k = 0; k < objects.size(); ++k) {
    //     auto object = objects[k];
    for (uint32_t k = 0; k < num_meshes; ++k) {
        auto object = meshes_data[k];
        if (object->material.type == Material::Type::Light) {
            emit_area_sum += object->getArea();
            if (p <= emit_area_sum) {
                object->Sample(pos, pdf);
                break;
            }
        }
    }
}

float Scene::lightAreaSum() const {
    float emit_area_sum = 0;
    // for (uint32_t k = 0; k < objects.size(); ++k) {
    //     auto object = objects[k];
    for (uint32_t k = 0; k < num_meshes; ++k) {
        auto object = meshes_data[k];
        if (object->material.type == Material::Type::Light) {
            emit_area_sum += object->getArea();
        }
    }
    return emit_area_sum;
}

glm::vec3 Scene::castRay(const Ray &eyeRay) const {
    glm::vec3 accRadiance(0.f), throughput(1.f);

    Ray ray = eyeRay;
    Intersection intersec = Scene::intersect(ray);
    Material material = intersec.m;

    if (!intersec.happened) {
        // sample envmap
        return accRadiance;
    }

    if (intersec.m.type == Material::Type::Light) {
        accRadiance = material.baseColor;
        return accRadiance;
    }
    for (int depth = 1; depth <= this->maxDepth; depth++) {
		bool deltaBSDF = (material.type == Material::Type::Dielectric); // ? why this

        if (!deltaBSDF) {
            // direct lighting
            Intersection lightSample;
            float lightPdf;
            sampleLight(lightSample, lightPdf);
            glm::vec3 p = intersec.coords;
            glm::vec3 x = lightSample.coords;
            glm::vec3 wo = -ray.direction;
            glm::vec3 wi = glm::normalize(x - p);
            Ray shadowRay(p + wi * Epsilon5, wi);
            Intersection shadowIntersec = Scene::intersect(shadowRay);
            if (shadowIntersec.distance - glm::length(x - p) > -Epsilon5) {
                glm::vec3 emit = lightSample.emit;
                glm::vec3 f_r = material.BSDF(intersec.normal, wo, wi);
                float cos_theta = Math::satDot(intersec.normal, wi);
                float cos_theta_prime = Math::satDot(lightSample.normal, -wi);
                float r2 = glm::dot(x - p, x - p);
                accRadiance += throughput * emit * f_r * cos_theta * cos_theta_prime / r2 / lightPdf;
            }
        }

        // indirect lighting 
        BSDFSample sample;
        material.sample(intersec.normal, -ray.direction, Math::sample3D(), sample);
        glm::vec3 p = intersec.coords;
        glm::vec3 wo = -ray.direction;
        glm::vec3 wi = sample.dir;
        glm::vec3 f_r = sample.bsdf;
        float indirectPdf = sample.pdf;
        float cos_theta = Math::absDot(intersec.normal, wi);

        if (sample.type == BSDFSampleType::Invalid) {
            // Terminate path if sampling fails
            break;
        } else if (sample.pdf < Epsilon8) {
            break;
        }

        bool deltaSample = (sample.type & BSDFSampleType::Specular); // ? why this
        throughput *= f_r / indirectPdf * (deltaSample ? 1.f : cos_theta);

        // new ray, new intersection, new material
        ray = Ray(p + wi * Epsilon5, wi);
        intersec = Scene::intersect(ray);
        material = intersec.m;

        if (!intersec.happened) {
            // sample envmap
            break;
        }
        // if it is light again
        if (material.type == Material::Type::Light) {
            float lightPdf = Math::pdfAreaToSolidAngle(intersec.obj->getArea() / Scene::lightAreaSum(), p, intersec.coords, intersec.normal);
            float weight = deltaSample ? 1.f : Math::powerHeuristic(indirectPdf, lightPdf);
            accRadiance += throughput * intersec.emit * weight;
            break;
        }
    }
    return accRadiance;
}