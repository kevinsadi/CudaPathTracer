//
// Created by Göksu Güvendiren on 2019-05-14.
//
#include "Scene.hpp"
#include "Triangle.hpp"

void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const {
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const {
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()) {
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()) {
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum) {
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
    const Ray &ray,
    const std::vector<Object *> &objects,
    float &tNear, uint32_t &index, Object **hitObject) {
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
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
Vector3f Scene::castRay(const Ray &ray, int depth) const {
    Vector3f radiance(0);

    if (depth > this->maxDepth)
        return radiance;

    Intersection hit = Scene::intersect(ray);
    if (!hit.happened)
        return radiance;
    if (hit.m->hasEmission())
        return hit.m->getEmission();

    switch (hit.m->getType()) {
    case Lambert: {
        Vector3f L_dir(0.f), L_indir(0.f);
        // direct lighting
        Intersection light_sample;
        float light_pdf;
        sampleLight(light_sample, light_pdf);
        Vector3f p = hit.coords;
        Vector3f x = light_sample.coords;
        // outgoing radiance direction at p, which is typically the opposite of ray direction
        Vector3f wo = -ray.direction;
        // incident radiance direction at p, poting outwards
        Vector3f ws = (x - p).normalized();
        // shadow ray (I abuse the term in ray tracing), testing if this light is blocked
        Ray shadow_ray(p, ws);
        Intersection hit2 = Scene::intersect(shadow_ray);
        if (hit2.distance - (x - p).norm() > -epsilon) {
            // hit the light, no blockades
            Vector3f emit = light_sample.emit;
            Vector3f f_r = hit.m->eval(ws, wo, hit.normal);
            float cos_theta = std::max(0.0f, dotProduct(hit.normal, ws));
            float cos_theta_prime = std::max(0.0f, dotProduct(light_sample.normal, -ws));
            float r2 = dotProduct(x - p, x - p);
            L_dir = emit * f_r * cos_theta * cos_theta_prime / r2 / light_pdf;
        }
        // return L_dir; // for testing direct lighting only

        // indirect lighting
        if (get_random_float() < RussianRoulette) {
            Vector3f wi = hit.m->sample(wo, hit.normal).normalized();
            Ray ray_indir(p, wi);
            Intersection hit_indir = Scene::intersect(ray_indir);
            float pdf_indir = hit.m->pdf(wi, wo, hit.normal);
            // if the ray hits a non-emissive object, we can continue
            if (hit_indir.happened && !hit_indir.m->hasEmission() && pdf_indir > epsilon) {
                Vector3f f_r = hit.m->eval(wi, wo, hit.normal);
                float cos_theta = std::max(0.0f, dotProduct(hit.normal, wi));
                L_indir = Scene::castRay(ray_indir, depth + 1) * f_r * cos_theta / pdf_indir / RussianRoulette;
            }
        }
        radiance = L_dir + L_indir;
        break;
    }
    case Metal: {
        break;
    }
    case Dielectric: {
        break;
    }
    }

    return radiance;
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