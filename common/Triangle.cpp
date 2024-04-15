#include "Triangle.hpp"
#include "MathUtils.hpp"
#include "OBJ_Loader.hpp"

MeshTriangle::MeshTriangle(const std::string& filename, Material& mt) {
    objl::Loader loader;
    loader.LoadFile(filename);
    area = 0;
    material = mt;
    assert(loader.LoadedMeshes.size() == 1);
    auto mesh = loader.LoadedMeshes[0];

    // stored indexed mesh
    num_vertices = mesh.Vertices.size();
    vertices = new glm::vec3[num_vertices];
    stCoordinates = new glm::vec2[num_vertices];

    num_triangles = mesh.Indices.size() / 3;
    triangles = new Triangle[num_triangles];
    vertexIndex = new uint32_t[num_triangles * 3];
    glm::vec3 min_vert = glm::vec3{ std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity() };
    glm::vec3 max_vert = glm::vec3{ -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity() };
    int triangle_index = 0;
    for (int i = 0; i < mesh.Vertices.size(); i += 3) {
        std::array<glm::vec3, 3> face_vertices;

        for (int j = 0; j < 3; j++) {
            auto vert = glm::vec3(mesh.Vertices[i + j].Position.X,
                mesh.Vertices[i + j].Position.Y,
                mesh.Vertices[i + j].Position.Z);
            face_vertices[j] = vert;

            min_vert =
                glm::vec3(std::min(min_vert.x, vert.x), std::min(min_vert.y, vert.y),
                    std::min(min_vert.z, vert.z));
            max_vert =
                glm::vec3(std::max(max_vert.x, vert.x), std::max(max_vert.y, vert.y),
                    std::max(max_vert.z, vert.z));

            vertices[i + j] = vert;
            stCoordinates[i + j] = glm::vec2(mesh.Vertices[i + j].TextureCoordinate.X,
                mesh.Vertices[i + j].TextureCoordinate.Y);
        }

        // stored triangle soup
        triangles[triangle_index] = Triangle(face_vertices[0], face_vertices[1], face_vertices[2], mt);
        // kevin change
        triangles[triangle_index].t0 = stCoordinates[i];
        triangles[triangle_index].t1 = stCoordinates[i+1];
        triangles[triangle_index].t2 = stCoordinates[i+2];

        if (stCoordinates[i].x > 0) {
            std::cout << stCoordinates[i].x << std::endl;
        }

        area += triangles[triangle_index].area;
        ++triangle_index;
    }
    for (int i = 0; i < mesh.Indices.size(); i += 3) {
        vertexIndex[i] = mesh.Indices[i];
        vertexIndex[i + 1] = mesh.Indices[i + 1];
        vertexIndex[i + 2] = mesh.Indices[i + 2];
    }

    bounding_box = Bounds3(min_vert, max_vert);
}

MeshTriangle::~MeshTriangle() {
    if (vertices) delete[] vertices;
    if (vertexIndex) delete[] vertexIndex;
    if (stCoordinates) delete[] stCoordinates;
}