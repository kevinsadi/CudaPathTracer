#include "Renderer.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"
#include "Vector.hpp"
#include "global.hpp"
#include <chrono>

// In the main function of the program, we create the scene (create objects and
// lights) as well as set the options for the render (image width and height,
// maximum recursion depth, field-of-view, etc.). We then call the render
// function().
int main(int argc, char **argv) {
    int spp = 32;
    int maxDepth = 10;
    // read SPP & maxDepth from command line
    if (argc > 1)
        spp = atoi(argv[1]);
    if (argc > 2)
        maxDepth = atoi(argv[2]);

    // Change the definition here to change resolution
    // Scene scene(784, 784);
    Scene scene(512, 512);
    scene.maxDepth = maxDepth;

    Material *red = new Material(Lambert);
    red->m_albedo = Vector3f(0.63f, 0.065f, 0.05f);
    Material *green = new Material(Lambert);
    green->m_albedo = Vector3f(0.14f, 0.45f, 0.091f);
    Material *white = new Material(Lambert);
    white->m_albedo = Vector3f(0.725f, 0.71f, 0.68f);
    Material *light = new Material(Lambert, (8.0f * Vector3f(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * Vector3f(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * Vector3f(0.737f + 0.642f, 0.737f + 0.159f, 0.737f)));
    light->m_albedo = Vector3f(0.65f);

    MeshTriangle floor("../../models/cornellbox/floor.obj", white);
    MeshTriangle shortbox("../../models/cornellbox/shortbox.obj", white);
    MeshTriangle tallbox("../../models/cornellbox/tallbox.obj", white);
    MeshTriangle left("../../models/cornellbox/left.obj", red);
    MeshTriangle right("../../models/cornellbox/right.obj", green);
    MeshTriangle light_("../../models/cornellbox/light.obj", light);

    scene.Add(&floor);
    scene.Add(&shortbox);
    scene.Add(&tallbox);
    scene.Add(&left);
    scene.Add(&right);
    scene.Add(&light_);

    scene.buildBVH();

#if defined(_OPENMP)
    std::cout << "OpenMP: Enabled\n";
#else
    std::cout << "OpenMP: Disabled\n";
#endif

    Renderer r;
    r.spp = spp;

    auto start = std::chrono::system_clock::now();
    r.Render(scene);
    auto stop = std::chrono::system_clock::now();

    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds\n";

    return 0;
}