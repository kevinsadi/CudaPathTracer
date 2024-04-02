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

bool Scene::trace(
    const Ray& ray,
    const std::vector<Object*>& objects,
    float& tNear, uint32_t& index, Object** hitObject) {
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kFloatInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }

    return (*hitObject != nullptr);
}

static const double epsilon = 0.0005;

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

        ray = Ray(intersec.coords, fromGlm(sampleDir), epsilon);

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

Scene Scene::CreateBuiltinScene(Scene::BuiltinScene sceneId, int maxDepth)
{
    if (sceneId == Scene::BuiltinScene::CornellBox)
    {
        Scene scene(512, 512);
        scene.name = "Cornell Box";
        scene.maxDepth = maxDepth;
        scene.camPos = Vector3f(278, 273, -800);

        Material* red = new Material(Lambert);
        red->m_albedo = Vector3f(0.63f, 0.065f, 0.05f);
        Material* green = new Material(Lambert);
        green->m_albedo = Vector3f(0.14f, 0.45f, 0.091f);
        Material* white = new Material(Lambert);
        white->m_albedo = Vector3f(0.725f, 0.71f, 0.68f);
        Material* light = new Material(Lambert, (8.0f * Vector3f(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * Vector3f(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * Vector3f(0.737f + 0.642f, 0.737f + 0.159f, 0.737f)));
        light->m_albedo = Vector3f(0.65f);

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