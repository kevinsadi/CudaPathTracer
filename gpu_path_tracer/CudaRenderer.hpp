#include <common/Renderer.hpp>
#include <common/Scene.hpp>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include "CudaPathTrace.h"

enum CudaRenderMode
{
    SingleKernel = 1,
    Streamed = 2
};

class CudaRenderer : public Renderer {
private:
    CudaRenderMode mode = CudaRenderMode::SingleKernel;
    Scene* scene_gpu = nullptr;
    thrust::device_ptr<glm::vec3> framebuffer_gpu;
    thrust::device_ptr<PathSegment> pathSegments;
    thrust::device_ptr<PathSegment> termPathSegments;
    thrust::device_ptr<Intersection> intersections;
    int num_blocks = 0;
    int num_pixels = 0;
public:
    void SetMode(CudaRenderMode mode);
    void PrepareRender(const Scene& scene);
    void Render(const Scene& scene) override;
    void FinishRender(const Scene& scene);

    // this compute one batch/frame of path tracing, call it many times to get targeted spp
    void PathTrace(const Scene& scene, int iter);
};