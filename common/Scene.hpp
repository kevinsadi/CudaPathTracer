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
    float emit_area_sum = 0;
    // for (uint32_t k = 0; k < objects.size(); ++k) {
    //     auto object = objects[k];
    for (uint32_t k = 0; k < num_meshes; ++k) {
        auto object = meshes_data[k];
        if (object->material.type == Material::Type::Light) {
            emit_area_sum += object->getArea();
        }
    }
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
    for (uint32_t k = 0; k < num_meshes; ++k) {
        auto object = meshes_data[k];
        if (object->material.type == Material::Type::Light) {
            emit_area_sum += object->getArea();
        }
    }
    return emit_area_sum;
}

// Implementation of Path Tracing
glm::vec3 Scene::castRay(const Ray &eyeRay) const {
    glm::vec3 accRadiance(0.f);

    Ray ray = eyeRay;
    Intersection intersec = Scene::intersect(ray);

    // if not hit anything, return background color
    if (!intersec.happened) {
        // if (scene->envMap != nullptr) {
        // 	glm::vec2 uv = Math::toPlane(ray.direction);
        // 	accRadiance += scene->envMap->linearSample(uv);
        // }
        return accRadiance;
    }

    // TODO sample textured material
    Material &material = intersec.m; // scene->getTexturedMaterialAndSurface(intersec);
    // hit a light, return light color
    if (material.type == Material::Type::Light) {
        if (glm::dot(intersec.normal, ray.direction) > 0.f) {
            accRadiance = material.baseColor;
        }
        return accRadiance;
    }
    glm::vec3 throughput(1.f);
    glm::vec3 wo = -ray.direction;

    for (int depth = 1; depth <= this->maxDepth; depth++) {
        bool deltaBSDF = (material.type == Material::Type::Dielectric);

        if (material.type != Material::Type::Dielectric &&
            glm::dot(intersec.normal, wo) < 0.f) {
            intersec.normal = -intersec.normal;
        }

        if (!deltaBSDF) {
            glm::vec3 radiance; // emitted radiance
            glm::vec3 wi;

            Intersection lightSample;
            float lightPdf;
            sampleLight(lightSample, lightPdf);
            // float lightPdf = scene->sampleDirectLight(intersec.pos, sample4D(rng),
            // radiance, wi);
            radiance = lightSample.emit;
            wi = glm::normalize(lightSample.coords - intersec.coords);

            if (lightPdf > 0) {
                float BSDFPdf = material.pdf(intersec.normal, wo, wi);
                accRadiance += throughput * material.BSDF(intersec.normal, wo, wi) *
                               radiance * Math::satDot(intersec.normal, wi) / lightPdf *
                               Math::powerHeuristic(lightPdf, BSDFPdf);
            }
        }
        // indirect lighting
        BSDFSample sample;
        material.sample(intersec.normal, wo, Math::sample3D(), sample);

        if (sample.type == BSDFSampleType::Invalid) {
            // Terminate path if sampling fails
            break;
        } else if (sample.pdf < Epsilon8) {
            break;
        }
        bool deltaSample = (sample.type & BSDFSampleType::Specular);
        throughput *= sample.bsdf / sample.pdf *
                      (deltaSample ? 1.f : Math::absDot(intersec.normal, sample.dir));

        ray = Ray(intersec.coords + sample.dir * Epsilon5, sample.dir);
        // ray = makeOffsetedRay(intersec.coords, sample.dir);

        glm::vec3 curPos = intersec.coords;
        intersec = Scene::intersect(ray);
        wo = -ray.direction;

        if (!intersec.happened) {
            // if (scene->envMap != nullptr) {
            //     glm::vec3 radiance = scene->envMap->linearSample(Math::toPlane(ray.direction)) * throughput;
            //     float weight = deltaSample ? 1.f : Math::powerHeuristic(sample.pdf, scene->environmentMapPdf(ray.direction));
            //     accRadiance += radiance * weight;
            // }
            break;
        }
        // TODO sample textured material
        material = intersec.m; // material = scene->getTexturedMaterialAndSurface(intersec);

        if (material.type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
            if (glm::dot(intersec.normal, ray.direction) < 0.f) {
                break;
            }
#endif
            glm::vec3 radiance = material.baseColor;

            float lightPdf = Math::pdfAreaToSolidAngle(intersec.obj->getArea() / Scene::lightAreaSum(), curPos, intersec.coords, intersec.normal);

            float weight = deltaSample ? 1.f : Math::powerHeuristic(sample.pdf, lightPdf);
            accRadiance += radiance * throughput * weight;
            break;
        }
    }

    return accRadiance;
}