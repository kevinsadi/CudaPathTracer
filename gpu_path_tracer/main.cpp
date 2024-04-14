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

    auto start = std::chrono::system_clock::now();
    r.Render(scene);
    auto stop = std::chrono::system_clock::now();

    Utility::SavePPM("out/gpu/" + scene.name + ".ppm", r.framebuffer, scene.width, scene.height);

    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f << " seconds\n";

    return 0;
}