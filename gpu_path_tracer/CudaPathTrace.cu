#include <cstdio>
#include "CudaPathTrace.h"
#include "CudaUtils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
// #include <thrust/sort.h>

__global__ void SingleKernelRayTracing(Scene* scene_gpu, glm::vec3* framebuffer, int iter, int spp)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= (scene_gpu->width * scene_gpu->height)) return;
    float scale = tan(glm::radians(scene_gpu->fov * 0.5));
    float imageAspectRatio = scene_gpu->width / (float)scene_gpu->height;
    glm::vec3& eye_pos = scene_gpu->camPos;

    int i = tid % scene_gpu->width;
    int j = tid / scene_gpu->width;

    // jitter sampling for anti-aliasing
    RNG rng = RNG(iter, tid, 0, nullptr);
    float bias = rng.sample1D();
    float x = (2 * (i + bias) / (float)scene_gpu->width - 1) * imageAspectRatio * scale;
    float y = (1 - 2 * (j + bias) / (float)scene_gpu->height) * scale;
    glm::vec3 dir = normalize(glm::vec3(-x, y, 1));
    framebuffer[tid] += scene_gpu->castRay(rng, Ray(eye_pos, dir)) / (spp * 1.0f);
}

__global__ void GenerateCameraRay(Scene* scene_gpu, PathSegment* pathSegments, int iter, int maxDepth)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= (scene_gpu->width * scene_gpu->height)) return;

    float scale = tan(glm::radians(scene_gpu->fov * 0.5));
    float imageAspectRatio = scene_gpu->width / (float)scene_gpu->height;
    glm::vec3& eye_pos = scene_gpu->camPos;

    int i = tid % scene_gpu->width;
    int j = tid / scene_gpu->width;

    // jitter sampling for anti-aliasing
    RNG rng(iter, tid, 0, nullptr);
    float bias = rng.sample1D();
    float x = (2 * (i + bias) / (float)scene_gpu->width - 1) * imageAspectRatio * scale;
    float y = (1 - 2 * (j + bias) / (float)scene_gpu->height) * scale;
    glm::vec3 dir = normalize(glm::vec3(-x, y, 1));
    PathSegment pathSegment(
        rng,
        Ray(eye_pos, dir),
        glm::vec3(1.0f),
        glm::vec3(0.0f),
        tid,
        maxDepth
    );

    pathSegments[tid] = pathSegment;
}

__global__ void ComputeIntersections(int depth,
    int numPaths,
    Scene* scene_gpu,
    PathSegment* pathSegments,
    Intersection* intersections)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numPaths) return;

    PathSegment& pathSegment = pathSegments[tid];
    // Intersection intersec = scene_gpu->intersect(pathSegment.ray);
    pathSegment.remainingBounces -= 1;
    // intersections[tid] = intersec;
    intersections[tid] = scene_gpu->intersect(pathSegment.ray);
}

__global__ void IntegratePathSegment(Scene* scene_gpu, PathSegment* pathSegments, Intersection* intersections, int numPaths)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numPaths) return;

    PathSegment& pathSegment = pathSegments[tid];
    Intersection& intersec = intersections[tid];
    Material& material = intersec.m;
    RNG& rng = pathSegment.rng;
    Ray& ray = pathSegment.ray;
    glm::vec3& throughput = pathSegment.throughput;
    glm::vec3& accRadiance = pathSegment.radiance;

    // sanity check
    if (!intersec.happened)
    {
        // sample envmap
        pathSegment.remainingBounces = 0;
        return;
    }
    if (material.emitting())
    {
        // stop bouncing
        // pathSegment.radiance = material.emission();
        pathSegment.remainingBounces = 0;
        return;
    }

    // compute direct lighting
    if (material.type != Material::Type::Glass) {
        // importance sampling: instead of sampling all ray directions - sampling over all solid angles dw,
        // which lead to many rays wasted, we sample over all light sources - sampling over all light areas dA
        Intersection lightSample;
        float lightSamplePdf;
        scene_gpu->sampleLight(rng, lightSample, lightSamplePdf);
        glm::vec3 p = intersec.coords;
        glm::vec3 x = lightSample.coords;
        glm::vec3 wo = -ray.direction;
        glm::vec3 wi = glm::normalize(x - p);
        Ray shadowRay(p + wi * Epsilon5, wi);
        Intersection shadowIntersec = scene_gpu->intersect(shadowRay);
        if (shadowIntersec.distance - glm::length(x - p) > -Epsilon4) {
            glm::vec3 emit = lightSample.emit;
            glm::vec3 bsdf = material.bsdf(wi, wo, intersec.normal);
            float cos_theta = Math::satDot(intersec.normal, wi);
            float cos_theta_prime = Math::satDot(lightSample.normal, -wi);
            float r2 = glm::dot(x - p, x - p);
            // dw = dA * (cos_theta_prime / r2)
            accRadiance += throughput * emit * bsdf * cos_theta * cos_theta_prime / r2 / lightSamplePdf;
        }
    }

    // indirect lighting
    glm::vec3 wo = -ray.direction;
    glm::vec3 wi = material.sample(rng, wo, intersec.normal);
    glm::vec3 p = intersec.coords;
    glm::vec3 bsdf = material.bsdf(wi, wo, intersec.normal);
    float bsdfSamplePdf = material.pdf(wi, wo, intersec.normal);
    float cos_theta = Math::absDot(intersec.normal, wi);

    if (bsdfSamplePdf < Epsilon5) {
        pathSegment.remainingBounces = 0;
        return;
    }

    throughput *= bsdf * cos_theta / bsdfSamplePdf;
    ray = Ray(p + wi * Epsilon5, wi);
}

__global__ void FinalGather(glm::vec3* framebuffer_gpu, PathSegment* termPathSegments, int numPaths, int spp)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numPaths) return;

    PathSegment& pathSegment = termPathSegments[tid];
    framebuffer_gpu[pathSegment.pixelIndex] += pathSegment.radiance * (1.0f / spp);
}

struct CompactTerminatedPaths {
    __device__ bool operator() (const PathSegment& segment) {
        return !(segment.pixelIndex >= 0 && segment.remainingBounces <= 0);
    }
};

struct RemoveInvalidPaths {
    __device__ bool operator() (const PathSegment& segment) {
        return segment.pixelIndex < 0 || segment.remainingBounces <= 0;
    }
};

void StreamedPathTracing(
    int num_threads,
    Scene* scene_gpu,
    glm::vec3* framebuffer_gpu,
    Intersection* intersections,
    thrust::device_ptr<PathSegment> pathSegments,
    thrust::device_ptr<PathSegment> termPathSegments,
    int numPixels, int maxDepth, int iter, int spp)
{
    int num_blocks_total = ComputeNumBlocks(numPixels, num_threads);
    int numPaths = numPixels;
    auto termPaths = termPathSegments;

    // 1. generate eye-rays
    // loop until no rays in ray-pool
    GenerateCameraRay << <num_blocks_total, num_threads >> > (scene_gpu, thrust::raw_pointer_cast(pathSegments), iter, maxDepth);
    checkCUDAError("Streamed::GenerateCameraRay");
    // cudaDeviceSynchronize();

    bool terminated = false;
    while (!terminated)
    {
        // clean intersections
        cudaMemset(intersections, 0, numPixels * sizeof(Intersection));

        // tracing
        // int num_blocks_tracing = ComputeNumBlocks(numPaths, num_threads);
        int num_blocks_tracing = num_blocks_total;
        ComputeIntersections << <num_blocks_tracing, num_threads >> > (maxDepth, numPaths, scene_gpu, thrust::raw_pointer_cast(pathSegments), intersections);
        checkCUDAError("Streamed::ComputeIntersections");
        // cudaDeviceSynchronize();
        IntegratePathSegment << <num_blocks_tracing, num_threads >> > (scene_gpu, thrust::raw_pointer_cast(pathSegments), intersections, numPaths);
        checkCUDAError("Streamed::IntegratePathSegment");
        // cudaDeviceSynchronize();

        // stored terminated paths
        termPaths = thrust::remove_copy_if(pathSegments, pathSegments + numPaths, termPaths, CompactTerminatedPaths());
        auto end = thrust::remove_if(pathSegments, pathSegments + numPaths, RemoveInvalidPaths());
        numPaths = end - pathSegments;

        terminated = numPaths == 0;
        checkCUDAError("Streamed::Compaction");
    }

    FinalGather << <num_blocks_total, num_threads >> > (framebuffer_gpu, thrust::raw_pointer_cast(termPathSegments), numPixels, spp);
    checkCUDAError("Streamed::FinalGather");
    // cudaDeviceSynchronize();
}