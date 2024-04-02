#include "CudaRenderer.hpp"
#include "CudaPathTrace.h"
#include <common/MathUtils.hpp>
#include <glm/glm.hpp>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#define NUM_THREADS 256

__global__ void SingleKernelRayTracing(Scene* scene_gpu, curandState* rng_gpu, Vector3f* framebuffer, int spp)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= scene_gpu->width * scene_gpu->height) return;

    float scale = tan(glm::radians(scene_gpu->fov * 0.5));
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
        framebuffer[tid] += scene_gpu->castRay(Ray(eye_pos, dir)) / spp;
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
    int seed = 0; // todo
    SetupRandomStates << <num_blocks, NUM_THREADS >> > (seed, rng_gpu, num_pixels);

    scene_gpu = nullptr;
    scene.MallocCuda(scene_gpu);

    framebuffer_gpu = thrust::device_malloc<Vector3f>(num_pixels);
}

void CudaRenderer::FinishRender(const Scene& scene)
{
    cudaFree(rng_gpu);
    rng_gpu = nullptr;

    scene.FreeCuda();
    scene_gpu = nullptr;

    thrust::device_free(framebuffer_gpu);
}

void CudaRenderer::Render(const Scene& scene)
{
    framebuffer = std::vector<Vector3f>(scene.width * scene.height);

    std::cout << "SPP: " << this->spp << "\n";
    // 1. clear framebuffer on gpu
    thrust::fill(framebuffer_gpu, framebuffer_gpu + num_pixels, Vector3f(0, 0, 0));
    // 2. render
    SingleKernelRayTracing << <num_blocks, NUM_THREADS >> > (scene_gpu, rng_gpu, thrust::raw_pointer_cast(framebuffer_gpu), this->spp);
    // 3. copy framebuffer from gpu to cpu
    cudaMemcpy(framebuffer.data(), thrust::raw_pointer_cast(framebuffer_gpu), num_pixels * sizeof(Vector3f), cudaMemcpyDeviceToHost);
}