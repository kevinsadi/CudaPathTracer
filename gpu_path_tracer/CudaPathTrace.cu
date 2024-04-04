// #include <cuda.h>
// #include <common/Scene.hpp>

// void SingleKernelRayTracing(Scene* scene_gpu, )
// {
//     framebuffer = std::vector<glm::vec3>(scene.width * scene.height);

//     float scale = tan(deg2rad(scene.fov * 0.5));
//     float imageAspectRatio = scene.width / (float)scene.height;
//     glm::vec3 eye_pos = scene.camPos;
//     int m = 0;

//     // change the spp value to change sample ammount
//     std::cout << "SPP: " << this->spp << "\n";
//     for (uint32_t j = 0; j < scene.height; ++j) {
//         for (uint32_t i = 0; i < scene.width; ++i) {
//             // generate primary ray direction
//             for (int k = 0; k < this->spp; k++) {
//                 // jitter sampling for anti-aliasing
//                 float bias = get_random_float();
//                 float x = (2 * (i + bias) / (float)scene.width - 1) * imageAspectRatio * scale;
//                 float y = (1 - 2 * (j + bias) / (float)scene.height) * scale;
//                 glm::vec3 dir = normalize(glm::vec3(-x, y, 1));
//                 framebuffer[m] += scene.castRay(Ray(eye_pos, dir), 0) / this->spp;
//             }
//             m++;
//         }
//     }
// }