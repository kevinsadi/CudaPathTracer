#include <common/Renderer.hpp>
#include <common/Scene.hpp>
// #include <common/Sphere.hpp>
#include <common/Triangle.hpp>
#include <common/Utility.hpp>
#include <chrono>

#if defined(_OPENMP)
#include <omp.h>
#endif

// In the main function of the program, we create the scene (create objects and
// lights) as well as set the options for the render (image width and height,
// maximum recursion depth, field-of-view, etc.). We then call the render
// function().
int main(int argc, char** argv) {
    int spp = 32;
    int maxDepth = 8;
    int omp_num_threads = 8;
    // read SPP & maxDepth from command line
    if (argc > 1)
        spp = atoi(argv[1]);
    if (argc > 2)
        maxDepth = atoi(argv[2]);
    if (argc > 3)
    {
        omp_num_threads = atoi(argv[3]);
    }

    // Change the definition here to change resolution
    Scene scene = Scene::CreateBuiltinScene(Scene::CornellBox, maxDepth);

#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(omp_num_threads);
    std::cout << "OpenMP: Enabled, #threads=" << omp_get_num_threads() << "\n";
#else
    std::cout << "OpenMP: Disabled\n";
#endif

    Renderer r;
    r.spp = spp;

    auto start = std::chrono::system_clock::now();
    r.Render(scene);
    auto stop = std::chrono::system_clock::now();

    Utility::SavePPM("out/cpu/" + scene.name + ".ppm", r.framebuffer, scene.width, scene.height);

    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f << " seconds\n";

    return 0;
}