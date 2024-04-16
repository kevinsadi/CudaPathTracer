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
    this->bvh = new BVHAccel(objects, meshBvhMap, 1, BVHAccel::SplitMethod::NAIVE);
    if (this->mesh_bvhs) {
        delete[] this->mesh_bvhs;
    }
    this->mesh_bvhs = new BVHAccel*[num_meshes];
    for (int i = 0; i < num_meshes; i++) {
        this->mesh_bvhs[i] = meshBvhMap[meshes[i]];
    }
}

Scene Scene::CreateBuiltinScene(Scene::BuiltinScene sceneId, int maxDepth)
{
    if (sceneId == Scene::BuiltinScene::CornellBox)
    {
        Scene scene(512, 512);
        scene.name = "Cornell Box";
        scene.maxDepth = maxDepth;
        scene.camPos = glm::vec3(278, 273, -800);

        Material red;
        red._albedo = glm::vec3(0.63f, 0.065f, 0.05f);
        Material green;
        green._albedo = glm::vec3(0.14f, 0.45f, 0.091f);
        Material white;
        white._albedo = glm::vec3(0.725f, 0.71f, 0.68f);
        Material light;
        light._emission = (8.0f * glm::vec3(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * glm::vec3(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * glm::vec3(0.737f + 0.642f, 0.737f + 0.159f, 0.737f));
        
        Material metal;
        metal.type = Material::Type::MetallicWorkflow;
        metal._albedo = glm::vec3(0.7, 0.5, 0.2);
        metal._metallic = 1.f;
        metal._roughness = 0.0050;

        Material glass;
        glass.type = Material::Type::Glass;
        glass._roughness = 0.002f;
        glass._ior = 1.5f;


        auto floor = new MeshTriangle("models/cornellbox/floor.obj", white);
        auto shortbox = new MeshTriangle("models/cornellbox/shortbox.obj", glass);
        auto tallbox = new MeshTriangle("models/cornellbox/tallbox.obj", metal);
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
    else if (sceneId == Scene::BuiltinScene::TextureTest) {
        Scene scene(512, 512);
        scene.name = "TextureTest";
        scene.maxDepth = maxDepth;
        scene.camPos = glm::vec3(278, 273, -800);

        Material red;
        red._albedo = glm::vec3(0.63f, 0.065f, 0.05f);
        red._ior = 0.f;
        Material green;
        green._albedo = glm::vec3(0.14f, 0.45f, 0.091f);
        green._ior = 0.f;
        Material white;
        white._albedo = glm::vec3(0.725f, 0.71f, 0.68f);
        white._ior = 0.f;
        const std::string woodTexture("textures/wood_floor_diff_4k.jpg");
        Image* image = new Image(woodTexture); 
        scene.mTextures.push_back(new Image(woodTexture));
        white.baseColorMapId = 0;

        Material light;
        light._emission = (8.0f * glm::vec3(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * glm::vec3(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * glm::vec3(0.737f + 0.642f, 0.737f + 0.159f, 0.737f));

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

        return scene; // RVO will optimize this	        return scene; // RVO will optimize this
    }
    else
    {
        throw std::runtime_error("Unsupported sceneId");
    }
}