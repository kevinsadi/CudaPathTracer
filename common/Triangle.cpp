#include "Triangle.hpp"
#include "OBJ_Loader.hpp"

MeshTriangle::MeshTriangle(const std::string& filename, Material* mt) {
    objl::Loader loader;
    loader.LoadFile(filename);
    area = 0;
    material = mt;
    assert(loader.LoadedMeshes.size() == 1);
    auto mesh = loader.LoadedMeshes[0];

    // stored indexed mesh
    num_vertices = mesh.Vertices.size();
    vertices = new Vector3f[num_vertices];
    stCoordinates = new Vector2f[num_vertices];

    num_triangles = mesh.Indices.size() / 3;
    triangles = new Triangle[num_triangles];
    vertexIndex = new uint32_t[num_triangles * 3];
    Vector3f min_vert = Vector3f{ std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity() };
    Vector3f max_vert = Vector3f{ -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity() };
    int triangle_index = 0;
    for (int i = 0; i < mesh.Vertices.size(); i += 3) {
        std::array<Vector3f, 3> face_vertices;

        for (int j = 0; j < 3; j++) {
            auto vert = Vector3f(mesh.Vertices[i + j].Position.X,
                mesh.Vertices[i + j].Position.Y,
                mesh.Vertices[i + j].Position.Z);
            face_vertices[j] = vert;

            min_vert =
                Vector3f(std::min(min_vert.x, vert.x), std::min(min_vert.y, vert.y),
                    std::min(min_vert.z, vert.z));
            max_vert =
                Vector3f(std::max(max_vert.x, vert.x), std::max(max_vert.y, vert.y),
                    std::max(max_vert.z, vert.z));

            vertices[i + j] = vert;
            stCoordinates[i + j] = Vector2f(mesh.Vertices[i + j].TextureCoordinate.X,
                mesh.Vertices[i + j].TextureCoordinate.Y);
        }

        triangles[triangle_index++] = Triangle(face_vertices[0], face_vertices[1], face_vertices[2], mt);
    }
    for (int i = 0; i < mesh.Indices.size(); i += 3) {
        vertexIndex[i] = mesh.Indices[i];
        vertexIndex[i + 1] = mesh.Indices[i + 1];
        vertexIndex[i + 2] = mesh.Indices[i + 2];
    }

    bounding_box = Bounds3(min_vert, max_vert);

    // stored triangle soup
    std::vector<Object*> triangle_ptrs; // temporary storage for BVH construction
    for (int i = 0; i < num_triangles; ++i) {
        triangle_ptrs.push_back(&triangles[i]);
        area += triangles[i].area;
    }

    bvh = new BVHAccel(triangle_ptrs);
}

MeshTriangle::~MeshTriangle() {
    if (vertices) delete[] vertices;
    if (vertexIndex) delete[] vertexIndex;
    if (stCoordinates) delete[] stCoordinates;
    delete bvh;
}