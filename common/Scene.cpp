//
// Created by Göksu Güvendiren on 2019-05-14.
//
#include "Scene.hpp"
#include "Triangle.hpp"
#include "Material.hpp"
#include "MathUtils.hpp"

void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    std::vector<Object*> objects;
    for (auto& mesh : meshes) {
        objects.push_back(mesh);
    }
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
    // this->bvh = new BVHAccel(meshes, 1, BVHAccel::SplitMethod::NAIVE);
}

// bool Scene::trace(
//     const Ray& ray,
//     const std::vector<Object*>& objects,
//     float& tNear, uint32_t& index, Object** hitObject) {
//     *hitObject = nullptr;
//     for (uint32_t k = 0; k < objects.size(); ++k) {
//         float tNearK = kFloatInfinity;
//         uint32_t indexK;
//         glm::vec2 uvK;
//         if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
//             *hitObject = objects[k];
//             tNear = tNearK;
//             index = indexK;
//         }
//     }

//     return (*hitObject != nullptr);
// }

Scene Scene::CreateBuiltinScene(Scene::BuiltinScene sceneId, int maxDepth)
{
    if (sceneId == Scene::BuiltinScene::CornellBox)
    {
        Scene scene(512, 512);
        scene.name = "Cornell Box";
        scene.maxDepth = maxDepth;
        scene.camPos = glm::vec3(278, 273, -800);

        // Material* red = new Material(Lambert);
        // red->m_albedo = glm::vec3(0.63f, 0.065f, 0.05f);
        // Material* green = new Material(Lambert);
        // green->m_albedo = glm::vec3(0.14f, 0.45f, 0.091f);
        // Material* white = new Material(Lambert);
        // white->m_albedo = glm::vec3(0.725f, 0.71f, 0.68f);
        // Material* light = new Material(Lambert, (8.0f * glm::vec3(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * glm::vec3(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * glm::vec3(0.737f + 0.642f, 0.737f + 0.159f, 0.737f)));
        // light->m_albedo = glm::vec3(0.65f);
        Material red;
        red.baseColor = glm::vec3(0.63f, 0.065f, 0.05f);
        red.ior = 0.f;
        Material green;
        green.baseColor = glm::vec3(0.14f, 0.45f, 0.091f);
        green.ior = 0.f;
        Material white;
        white.baseColor = glm::vec3(0.725f, 0.71f, 0.68f);
        white.ior = 0.f;
        Material light;
        light.type = Material::Type::Light;
        light.baseColor = (8.0f * glm::vec3(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * glm::vec3(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * glm::vec3(0.737f + 0.642f, 0.737f + 0.159f, 0.737f));
        light.ior = 0.f;
        

        scene.AddMaterial(red);
        scene.AddMaterial(green);
        scene.AddMaterial(white);
        scene.AddMaterial(light);

        auto floor = new MeshTriangle("models/cornellbox/floor.obj", white);
        auto shortbox = new MeshTriangle("models/cornellbox/shortbox.obj", white);
        auto tallbox = new MeshTriangle("models/cornellbox/tallbox.obj", white);
        auto left = new MeshTriangle("models/cornellbox/left.obj", red);
        auto right = new MeshTriangle("models/cornellbox/right.obj", green);
        auto light_ = new MeshTriangle("models/cornellbox/light.obj", light);

        scene.Add(floor);
        scene.Add(shortbox);
        scene.Add(tallbox);
        scene.Add(left);
        scene.Add(right);
        scene.Add(light_);

        scene.buildBVH();

        return scene; // RVO will optimize this
    }
    else
    {
        throw std::runtime_error("Unsupported sceneId");
    }
}