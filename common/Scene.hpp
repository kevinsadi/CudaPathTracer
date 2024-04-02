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
    FUNC_QUALIFIER Vector3f castRay(const Ray& ray) const;
    FUNC_QUALIFIER void sampleLight(Intersection& pos, float& pdf) const;
    FUNC_QUALIFIER bool trace(const Ray& ray, const std::vector<Object*>& objects, float& tNear, uint32_t& index, Object** hitObject);
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