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

#include "Image.hpp"

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
    BVHAccel *bvh = nullptr;
    std::vector<Image*> mTextures;
    // -----------Editor Only-----------
    // [!] as polymorphic is not supported in CUDA, currently we only allow MeshTriangle
    std::vector<MeshTriangle *> meshes;
    std::unordered_map<MeshTriangle *, BVHAccel *> meshBvhMap;
    // ---------------------------------
    MeshTriangle **meshes_data = nullptr;
    BVHAccel **mesh_bvhs = nullptr;
    int num_meshes = 0;

    Scene(int w, int h) : width(w), height(h) {}

    void Add(MeshTriangle *mesh) {
        meshes.push_back((MeshTriangle *)mesh);
        num_meshes = meshes.size();
        meshes_data = meshes.data();
    }

    // const std::vector<Object *> &get_objects() const { return objects; }
    // const std::vector<std::unique_ptr<Light>> &get_lights() const { return
    // lights; }
    FUNC_QUALIFIER inline Intersection intersect(const Ray &ray) const;
    void buildBVH();
    FUNC_QUALIFIER inline glm::vec3 castRay(RNG &rng, const Ray &eyeRay) const;
    FUNC_QUALIFIER inline void sampleLight(RNG &rng, Intersection &pos, float &pdf) const;
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
        TextureTest
    };
    static Scene CreateBuiltinScene(BuiltinScene sceneId, int maxDepth);

    CUDA_PORTABLE(Scene);
};

Intersection Scene::intersect(const Ray &ray) const {
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(RNG &rng, Intersection &pos, float &pdf) const {
    float emit_area_sum = 0;
    // for (uint32_t k = 0; k < objects.size(); ++k) {
    //     auto object = objects[k];
    for (uint32_t k = 0; k < num_meshes; ++k) {
        auto mesh = meshes_data[k];
        if (mesh->material.emitting()) {
            emit_area_sum += mesh->area;
        }
    }

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
                pos.emit = mesh->material.emission();
                break;
            }
        }
    }
}

glm::vec3 Scene::castRay(RNG& rng, const Ray& eyeRay) const {
    glm::vec3 accRadiance(0.f), throughput(1.f);

    Ray ray = eyeRay;
    Intersection intersec = Scene::intersect(ray);
    Material material = intersec.m;

    if (!intersec.happened) {
        // sample envmap
        return accRadiance;
    }

    if (intersec.m.emitting()) {
        accRadiance = material._emission;
        return accRadiance;
    }
    for (int depth = 1; depth <= this->maxDepth; depth++) {
        // direct lighting
        if (material.type != Material::Type::Glass) {
            Intersection lightSample;
            float lightPdf;
            sampleLight(rng, lightSample, lightPdf);
            glm::vec3 p = intersec.coords;
            glm::vec3 x = lightSample.coords;
            glm::vec3 wo = -ray.direction;
            glm::vec3 wi = glm::normalize(x - p);
            Ray shadowRay(p + wi * Epsilon5, wi);
            Intersection shadowIntersec = Scene::intersect(shadowRay);
            if (shadowIntersec.distance - glm::length(x - p) > -Epsilon4) {
                glm::vec3 emit = lightSample.emit;
                glm::vec3 f_r = material.fr(wi, wo, intersec.normal);
                float cos_theta = Math::satDot(intersec.normal, wi);
                float cos_theta_prime = Math::satDot(lightSample.normal, -wi);
                float r2 = glm::dot(x - p, x - p);
                accRadiance += throughput * emit * f_r * cos_theta * cos_theta_prime / r2 / lightPdf;
            }
        }
        // indirect lighting
        glm::vec3 wo = -ray.direction;
        glm::vec3 wi = material.sample(rng.sample3D(), wo, intersec.normal);
        glm::vec3 p = intersec.coords;
        glm::vec3 f_r = material.fr(wi, wo, intersec.normal);
        float indirectPdf = material.pdf(wi, wo, intersec.normal);
        float cos_theta = Math::absDot(intersec.normal, wi);

        if (indirectPdf < Epsilon5) {
            break;
        }

        throughput *= f_r * cos_theta / indirectPdf;
        ray = Ray(p + wi * Epsilon5, wi);
        intersec = Scene::intersect(ray);
        material = intersec.m;

        if (!intersec.happened) {
            // sample envmap
            break;
        }
        // if it is light again
        if (material.emitting()) {
            break;
        }
    }
    return accRadiance;
}