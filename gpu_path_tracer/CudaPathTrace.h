#pragma once
#include <cuda.h>
#include <common/Ray.hpp>
#include <common/Scene.hpp>
#include <common/MathUtils.hpp>
#include <common/CudaPortable.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

#define NUM_THREADS 256

struct PathSegment {
    PathSegment() = default;
    FUNC_QUALIFIER inline PathSegment(const RNG& rng, const Ray& ray, const glm::vec3& throughput, const glm::vec3& radiance, int pixelIndex, int remainingBounces)
        : rng(rng), ray(ray), throughput(throughput), radiance(radiance), pixelIndex(pixelIndex), remainingBounces(remainingBounces) {}
    RNG rng;
    Ray ray;
    glm::vec3 throughput;
    glm::vec3 radiance;
    // PrevBSDFSampleInfo prev;
    int pixelIndex;
    int remainingBounces;
};


__global__ void SingleKernelRayTracing(Scene* scene_gpu, glm::vec3* framebuffer, int iter, int spp);
__global__ void GenerateCameraRay(Scene* scene_gpu, PathSegment* pathSegments, int iter, int traceDepth);
void StreamedPathTracing(
    Scene* scene_gpu,
    glm::vec3* framebuffer_gpu,
    Intersection* intersections,
    thrust::device_ptr<PathSegment> pathSegments,
    thrust::device_ptr<PathSegment> termPathSegments,
    int numPixels, int maxDepth, int iter, int spp, int num_blocks);