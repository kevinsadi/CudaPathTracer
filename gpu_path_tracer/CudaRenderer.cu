#include "CudaRenderer.hpp"
#include "CudaPathTrace.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <common/global.hpp>
#define NUM_THREADS 256

__global__ void SingleKernelRayTracing(Scene* scene_gpu, curandState* rng_gpu, Vector3f* framebuffer, int spp)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= scene_gpu->width * scene_gpu->height) return;

    float scale = tan(deg2rad(scene_gpu->fov * 0.5));
    float imageAspectRatio = scene_gpu->width / (float)scene_gpu->height;
    Vector3f& eye_pos = scene_gpu->camPos;

    int i = tid % scene_gpu->width;
    int j = tid / scene_gpu->width;
    for (int k = 0; k < spp; k++) {
        // jitter sampling for anti-aliasing
        float bias = curand_uniform(&rng_gpu[tid]);
        float x = (2 * (i + bias) / (float)scene_gpu->width - 1) * imageAspectRatio * scale;
        float y = (1 - 2 * (j + bias) / (float)scene_gpu->height) * scale;
        Vector3f dir = normalize(Vector3f(-x, y, 1));
        framebuffer[tid] = scene_gpu->castRay(Ray(eye_pos, dir), 0) / spp;
    }
}

__global__ void SetupRandomStates(int globalSeed, curandState* states, int numStates)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < numStates; i += stride)
    {
        curand_init(globalSeed, i, 0, &states[i]);
    }
}

void CudaRenderer::PrepareRender(const Scene& scene)
{
    rng_gpu = nullptr;
    num_blocks = (scene.width * scene.height + NUM_THREADS - 1) / NUM_THREADS;
    num_pixels = scene.width * scene.height;
    cudaMalloc(&rng_gpu, num_pixels * sizeof(curandState)); // todo: change to MT19937?
    SetupRandomStates << <num_blocks, NUM_THREADS >> > (GetRandomSeed(), rng_gpu, num_pixels);

    scene_gpu = nullptr;
    scene.MallocCuda(scene_gpu);

    framebuffer_gpu = nullptr;
    cudaMalloc(&framebuffer_gpu, scene.width * scene.height * sizeof(Vector3f));
}

void CudaRenderer::FinishRender(const Scene& scene)
{
    cudaFree(rng_gpu);
    rng_gpu = nullptr;

    scene.FreeCuda();
    scene_gpu = nullptr;

    cudaFree(framebuffer_gpu);
    framebuffer_gpu = nullptr;
}

void CudaRenderer::Render(const Scene& scene)
{
    framebuffer = std::vector<Vector3f>(scene.width * scene.height);

    std::cout << "SPP: " << this->spp << "\n";
    SingleKernelRayTracing << <num_blocks, NUM_THREADS >> > (scene_gpu, rng_gpu, framebuffer_gpu, this->spp);
}