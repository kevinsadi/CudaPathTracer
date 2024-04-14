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



void CudaRenderer::SetMode(CudaRenderMode mode)
{
    this->mode = mode;
}

void CudaRenderer::PrepareRender(const Scene& scene)
{
    num_blocks = (scene.width * scene.height + NUM_THREADS - 1) / NUM_THREADS;
    num_pixels = scene.width * scene.height;

    scene_gpu = nullptr;
    scene.MallocCuda(scene_gpu);

    framebuffer_gpu = thrust::device_malloc<glm::vec3>(num_pixels);
    if (mode == CudaRenderMode::Streamed)
    {
        pathSegments = thrust::device_malloc<PathSegment>(num_pixels);
        termPathSegments = thrust::device_malloc<PathSegment>(num_pixels);
        intersections = thrust::device_malloc<Intersection>(num_pixels);
    }
}

void CudaRenderer::FinishRender(const Scene& scene)
{
    scene.FreeCuda();
    scene_gpu = nullptr;

    thrust::device_free(framebuffer_gpu);
    if (mode == CudaRenderMode::Streamed)
    {
        thrust::device_free(pathSegments);
        thrust::device_free(termPathSegments);
        thrust::device_free(intersections);
    }
}

void CudaRenderer::Render(const Scene& scene)
{
    framebuffer = std::vector<glm::vec3>(scene.width * scene.height);

    PrepareRender(scene);
    {
        // 1. clear framebuffer on gpu
        thrust::fill(framebuffer_gpu, framebuffer_gpu + num_pixels, glm::vec3(0, 0, 0));
        // 2. render
        for (int i = 0; i < this->spp; ++i)
        {
            PathTrace(scene, i);
        }
        // 3. copy framebuffer from gpu to cpu
        cudaDeviceSynchronize();
        // thrust::copy(framebuffer_gpu, framebuffer_gpu + num_pixels, framebuffer.begin());
        cudaMemcpy(framebuffer.data(), thrust::raw_pointer_cast(framebuffer_gpu), num_pixels * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    FinishRender(scene);
}

void CudaRenderer::PathTrace(const Scene& scene, int iter)
{
    if (mode == CudaRenderMode::SingleKernel)
    {
        SingleKernelRayTracing << <num_blocks, NUM_THREADS >> > (scene_gpu, thrust::raw_pointer_cast(framebuffer_gpu), iter, this->spp);
    }
    else if (mode == CudaRenderMode::Streamed)
    {
        StreamedPathTracing(
            scene_gpu,
            thrust::raw_pointer_cast(framebuffer_gpu),
            thrust::raw_pointer_cast(intersections),
            pathSegments,
            termPathSegments,
            num_pixels, scene.maxDepth, iter, this->spp, num_blocks);
    }
}