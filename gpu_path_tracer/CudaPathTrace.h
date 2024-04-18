#pragma once
#include <cuda.h>
#include <common/Ray.hpp>
#include <common/Scene.hpp>
#include <common/MathUtils.hpp>
#include <common/CudaPortable.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>



__global__ void SingleKernelRayTracing(Scene* scene_gpu, glm::vec3* framebuffer, int iter, int spp);
__global__ void GenerateCameraRay(Scene* scene_gpu, PathSegment* pathSegments, int iter, int traceDepth);
void StreamedPathTracing(
    int num_threads,
    Scene* scene_gpu,
    glm::vec3* framebuffer_gpu,
    Intersection* intersections,
    thrust::device_ptr<PathSegment> pathSegments,
    thrust::device_ptr<PathSegment> termPathSegments,
    int numPixels, int maxDepth, int iter, int spp);


FUNC_QUALIFIER inline int ComputeNumBlocks(int total, int blockSize)
{
    return (total + blockSize - 1) / blockSize;
}