#include <common/Renderer.hpp>
#include <common/Scene.hpp>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

class CudaRenderer : public Renderer {
private:
    Scene* scene_gpu = nullptr;
    Vector3f* framebuffer_gpu = nullptr;
    curandState* rng_gpu = nullptr;
    int num_blocks = 0;
    int num_pixels = 0;
public:
    void PrepareRender(const Scene& scene);
    void Render(const Scene& scene) override;
    void FinishRender(const Scene& scene);
};