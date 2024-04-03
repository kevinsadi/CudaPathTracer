#include <common/Renderer.hpp>
#include <common/Scene.hpp>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

class CudaRenderer : public Renderer {
private:
    Scene* scene_gpu = nullptr;
    thrust::device_ptr<glm::vec3> framebuffer_gpu;
    int num_blocks = 0;
    int num_pixels = 0;
public:
    void PrepareRender(const Scene& scene);
    void Render(const Scene& scene) override;
    void FinishRender(const Scene& scene);
};