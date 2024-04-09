#include "CudaRenderer.hpp"
#include "CudaPathTrace.h"
#include <common/Scene.hpp>
#include <common/MathUtils.hpp>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#define NUM_THREADS 256

__global__ void SingleKernelRayTracing(Scene* scene_gpu, glm::vec3* framebuffer, int spp)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= (scene_gpu->width * scene_gpu->height)) return;
    float scale = tan(glm::radians(scene_gpu->fov * 0.5));
    float imageAspectRatio = scene_gpu->width / (float)scene_gpu->height;
    glm::vec3& eye_pos = scene_gpu->camPos;

    int i = tid % scene_gpu->width;
    int j = tid / scene_gpu->width;
    for (int k = 0; k < spp; k++) {
        // jitter sampling for anti-aliasing
        RNG rng = RNG(k, tid, 0, nullptr);
        float bias = rng.sample1D();
        float x = (2 * (i + bias) / (float)scene_gpu->width - 1) * imageAspectRatio * scale;
        float y = (1 - 2 * (j + bias) / (float)scene_gpu->height) * scale;
        glm::vec3 dir = normalize(glm::vec3(-x, y, 1));
        framebuffer[tid] += scene_gpu->castRay(rng, Ray(eye_pos, dir)) / (spp * 1.0f);
    }
}

void CudaRenderer::PrepareRender(const Scene& scene)
{
    num_blocks = (scene.width * scene.height + NUM_THREADS - 1) / NUM_THREADS;
    num_pixels = scene.width * scene.height;

    scene_gpu = nullptr;
    scene.MallocCuda(scene_gpu);

    framebuffer_gpu = thrust::device_malloc<glm::vec3>(num_pixels);
}

void CudaRenderer::FinishRender(const Scene& scene)
{
    scene.FreeCuda();
    scene_gpu = nullptr;

    thrust::device_free(framebuffer_gpu);
}

void CudaRenderer::Render(const Scene& scene)
{
    framebuffer = std::vector<glm::vec3>(scene.width * scene.height);

    PrepareRender(scene);
    {
        std::cout << "SPP: " << this->spp << "\n";
        // 1. clear framebuffer on gpu
        thrust::fill(framebuffer_gpu, framebuffer_gpu + num_pixels, glm::vec3(0, 0, 0));
        // 2. render
        SingleKernelRayTracing << <num_blocks, NUM_THREADS >> > (scene_gpu, thrust::raw_pointer_cast(framebuffer_gpu), this->spp);
        // 3. copy framebuffer from gpu to cpu
        cudaDeviceSynchronize();
        // thrust::copy(framebuffer_gpu, framebuffer_gpu + num_pixels, framebuffer.begin());
        cudaMemcpy(framebuffer.data(), thrust::raw_pointer_cast(framebuffer_gpu), num_pixels * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    FinishRender(scene);
}