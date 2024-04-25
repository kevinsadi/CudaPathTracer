//
// Created by goksu on 2/25/20.
//

#include "Scene.hpp"
#include "Renderer.hpp"
#include "MathUtils.hpp"
#include "Utility.hpp"
#include <omp.h>

inline float deg2rad(const float& deg) { return deg * Pi / 180.0; }

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
    framebuffer = std::vector<glm::vec3>(scene.width * scene.height);

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    glm::vec3 eye_pos = scene.camPos;
    int m = 0;

    // change the spp value to change sample ammount
    std::cout << "SPP: " << this->spp << "\n";
    std::cout << "Max Depth: " << scene.maxDepth << "\n";
    for (int k = 0; k < this->spp; k++){
#pragma omp parallel for collapse(2) private (m)
        for (uint32_t j = 0; j < scene.height; ++j) {
            for (uint32_t i = 0; i < scene.width; ++i) {
                int m = j * scene.width + i;
                // generate primary ray direction
                RNG rng = RNG(k, m, 0, nullptr);
                // jitter sampling for anti-aliasing
                float bias = rng.sample1D();
                float x = (2 * (i + bias) / (float)scene.width - 1) * imageAspectRatio * scale;
                float y = (1 - 2 * (j + bias) / (float)scene.height) * scale;
                glm::vec3 dir = glm::normalize(glm::vec3(-x, y, 1));
                framebuffer[m] += scene.castRay(rng, Ray(eye_pos, dir)) / (float)this->spp;  
            }
        }
#ifdef _OPENMP
        if (omp_get_thread_num() == 0)
#endif
        {
            Utility::UpdateProgress(k / (float)this->spp);
        }
    }
    Utility::UpdateProgress(1.f);
}
