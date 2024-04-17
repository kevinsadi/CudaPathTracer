#include "CudaRenderer.hpp"
#include <common/Scene.hpp>
#include <common/Utility.hpp>
#include <chrono>

// In the main function of the program, we create the scene (create objects and
// lights) as well as set the options for the render (image width and height,
// maximum recursion depth, field-of-view, etc.). We then call the render
// function().
int main(int argc, char** argv) {
    int spp = 32;
    int maxDepth = 10;
    CudaRenderMode mode = CudaRenderMode::SingleKernel;
    // read SPP & maxDepth from command line
    if (argc > 1)
        spp = atoi(argv[1]);
    if (argc > 2)
        maxDepth = atoi(argv[2]);
    if (argc > 3)
    {
        mode = (CudaRenderMode)atoi(argv[3]);
    }


    // Change the definition here to change resolution
    Scene scene = Scene::CreateBuiltinScene(Scene::CornellBox, maxDepth);

    // Log statistics
    std::cout << "Resolution: " << scene.width << "x" << scene.height << "\n";
    std::cout << "SPP: " << spp << "\n";
    std::cout << "Trace Depth: " << maxDepth << "\n";
    std::cout << "CUDA Mode: " << (
        mode == CudaRenderMode::SingleKernel ? "SingleKernel" 
                                                : "Streamed"
    )<< std::endl;

    CudaRenderer r;
    r.spp = spp;
    r.SetMode(mode);
    // r.SetMode(CudaRenderMode::Streamed);
    // r.SetMode(CudaRenderMode::SingleKernel);

    r.PrepareRender(scene);
    r.Render(scene);
    r.FinishRender(scene);

    Utility::SavePPM("out/gpu/" + scene.name + ".ppm", r.framebuffer, scene.width, scene.height);

    return 0;
}