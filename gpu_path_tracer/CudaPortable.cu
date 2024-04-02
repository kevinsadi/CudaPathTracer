#include "CudaPathTrace.h"
#include <common/CudaPortable.hpp>
#include <common/Vector.hpp>
#include <common/Ray.hpp>
#include <common/Material.hpp>
#include <common/Object.hpp>
#include <common/Triangle.hpp>
#include <common/Sphere.hpp>
#include <common/Scene.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>

// key: host ptr, value: device ptr
// a lookup table when tansfering host data to device
static std::unordered_map<void*, void*> memoryMap;
static std::unordered_map<void*, void*> memoryMap_inv;

static void* GetDeviceMemory(void* host_ptr)
{
    if (memoryMap.find(host_ptr) != memoryMap.end())
    {
        return memoryMap[host_ptr];
    }
    else
    {
        throw std::runtime_error("Object not found in objectMap: object being accessed is not uploaded to device memory");
    }
}
static void RegisterMemory(void* host_ptr, void* device_ptr)
{
    if (memoryMap.find(host_ptr) == memoryMap.end())
    {
        memoryMap[host_ptr] = device_ptr;
        memoryMap_inv[device_ptr] = host_ptr;
    }
    else
    {
        throw std::runtime_error("Object already exists in memoryMap: object being allocated is already uploaded to device memory");
    }
}
static void UnregisterMemory(void* host_ptr, void* device_ptr)
{
    if (memoryMap_inv.find(device_ptr) != memoryMap_inv.end())
    {
        memoryMap.erase(memoryMap_inv[device_ptr]);
        memoryMap_inv.erase(device_ptr);
    }
    else
    {
        throw std::runtime_error("Object not found in objectMap: object being deleted is not uploaded to device memory");
    }
}

template<typename T>
void AllocateAndRegister(const T* host, T*& device, int count)
{
    assert(host != nullptr);
    T** temp_device = nullptr;
    cudaMalloc(temp_device, sizeof(T) * count);
    device = *temp_device;
    RegisterMemory((void*)host, (void*)device);
}
template<typename T>
void AllocateAndRegisterIfNullptr(const T* host, T*& device, int count)
{
    if (device == nullptr)
    {
        AllocateAndRegister(host, device, count);
    }
    else
    {
        RegisterMemory((void*)host, (void*)device);
    }
}
template<typename T>
void FreeAndUnregister(const T* host, T* device)
{
    assert(host != nullptr);
    assert(device != nullptr);
    UnregisterMemory((void*)host, (void*)device);
    cudaFree(device);
}
template<typename T>
void FreeAndUnregister(const T* host)
{
    assert(host != nullptr);
    T* device = (T*)GetDeviceMemory((void*)host);
    FreeAndUnregister(host, device);
}


#ifdef GPU_PATH_TRACER
#define CUDA_AUTO_ALLOCATION(CLASS_NAME) \
    void CLASS_NAME::MallocCuda(CLASS_NAME*& device_ptr) const \
    {\
        AllocateAndRegisterIfNullptr(this, device_ptr, 1);\
        cudaMemcpy(device_ptr, this, sizeof(CLASS_NAME), cudaMemcpyHostToDevice);\
    }\
    void CLASS_NAME::FreeCuda() const \
    {\
        FreeAndUnregister(this);\
    }
#else
#define CUDA_AUTO_ALLOCATION(CLASS_NAME) 
#endif
// if class/struct is pure data type without pointer, then it is safe to use this macro
CUDA_AUTO_ALLOCATION(Vector3f);
CUDA_AUTO_ALLOCATION(Vector2f);
CUDA_AUTO_ALLOCATION(Material);
// otherwise, we need to manually implement the MallocCuda and FreeCuda function

void BVHBuildNode::MallocCuda(BVHBuildNode*& device_ptr) const
{
    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    BVHBuildNode temp = *this;
    if (this->left != nullptr)
    {
        temp.left = nullptr;
        this->left->MallocCuda(temp.left);
    }
    if (this->right != nullptr)
    {
        temp.right = nullptr;
        this->right->MallocCuda(temp.right);
    }
    if (this->object != nullptr)
    {
        temp.object = (Object*)GetDeviceMemory((void*)this->object);
    }
    cudaMemcpy(device_ptr, &temp, sizeof(BVHBuildNode), cudaMemcpyHostToDevice);
}
void BVHBuildNode::FreeCuda() const
{
    if (this->left != nullptr)
    {
        this->left->FreeCuda();
    }
    if (this->right != nullptr)
    {
        this->right->FreeCuda();
    }
    FreeAndUnregister(this);
}

void BVHAccel::MallocCuda(BVHAccel*& device_ptr) const
{
    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    BVHAccel temp = *this;
    if (this->root != nullptr)
    {
        temp.root = nullptr;
        this->root->MallocCuda(temp.root);
    }
    else
    {
        throw std::runtime_error("BVH root is null");
    }
    cudaMemcpy(device_ptr, &temp, sizeof(BVHAccel), cudaMemcpyHostToDevice);
}
void BVHAccel::FreeCuda() const
{
    if (this->root != nullptr)
    {
        this->root->FreeCuda();
    }
    else
    {
        throw std::runtime_error("BVH root is null");
    }
    FreeAndUnregister(this);
}

void Sphere::MallocCuda(Sphere*& device_ptr) const
{
    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    Sphere temp = *this;
    if (this->material != nullptr)
    {
        // not duplicate the material but point to the device memory of the material
        temp.material = (Material*)GetDeviceMemory(this->material);
    }
    cudaMemcpy(device_ptr, &temp, sizeof(Sphere), cudaMemcpyHostToDevice);
}
void Sphere::FreeCuda() const
{
    FreeAndUnregister(this);
}

void Triangle::MallocCuda(Triangle*& device_ptr) const
{
    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    Triangle temp = *this;
    if (temp.material != nullptr)
    {
        // not duplicate the material but point to the device memory of the material
        temp.material = (Material*)GetDeviceMemory(temp.material);
    }
    cudaMemcpy(device_ptr, &temp, sizeof(Triangle), cudaMemcpyHostToDevice);
}
void Triangle::FreeCuda() const
{
    FreeAndUnregister(this);
}

void MeshTriangle::MallocCuda(MeshTriangle*& device_ptr) const
{
    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    MeshTriangle temp = *this;
    assert(this->vertices != nullptr);
    assert(this->stCoordinates != nullptr);
    assert(this->vertexIndex != nullptr);
    assert(this->triangles != nullptr);
    assert(this->material != nullptr);
    assert(this->bvh != nullptr);
    // 1. index device material
    // not duplicate the material but point to the device memory of the material
    temp.material = (Material*)GetDeviceMemory(temp.material);
    // 2. upload indexed mesh data to device
    AllocateAndRegister(this->vertices, temp.vertices, temp.num_vertices);
    AllocateAndRegister(this->stCoordinates, temp.stCoordinates, temp.num_vertices);
    AllocateAndRegister(this->vertexIndex, temp.vertexIndex, temp.num_triangles * 3);
    cudaMemcpy(temp.vertices, this->vertices, sizeof(Vector3f) * temp.num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(temp.stCoordinates, this->stCoordinates, sizeof(Vector2f) * temp.num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(temp.vertexIndex, this->vertexIndex, sizeof(uint32_t) * temp.num_triangles * 3, cudaMemcpyHostToDevice);
    // 3. upload triangle soup to device
    AllocateAndRegister(this->triangles, temp.triangles, temp.num_triangles);
    for (int i = 0; i < temp.num_triangles; i++)
    {
        // allocate triangle on continuous memory
        Triangle* triangle_ptr = temp.triangles + i;
        this->triangles[i].MallocCuda(triangle_ptr);
    }
    // 4. allocate bvh
    temp.bvh = nullptr;
    this->bvh->MallocCuda(temp.bvh);

    cudaMemcpy(device_ptr, &temp, sizeof(MeshTriangle), cudaMemcpyHostToDevice);
}
void MeshTriangle::FreeCuda() const
{
    // 1. free indexed mesh data
    FreeAndUnregister(this->vertices);
    FreeAndUnregister(this->stCoordinates);
    FreeAndUnregister(this->vertexIndex);
    // 2. free triangle soup
    for (int i = 0; i < this->num_triangles; i++)
    {
        this->triangles[i].FreeCuda();
    }
    FreeAndUnregister(this->triangles);
    // 3. free bvh
    this->bvh->FreeCuda();

    FreeAndUnregister(this);
}

void Scene::MallocCuda(Scene*& device_ptr) const
{
    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    Scene temp = *this;
    assert(this->meshes_data != nullptr);
    assert(this->bvh != nullptr);
    // 1. upload bvh
    temp.bvh = nullptr;
    this->bvh->MallocCuda(temp.bvh);
    // 2. upload material
    AllocateAndRegister(this->materials_data, temp.materials_data, this->num_materials);
    Material** temp_materials = new Material * [this->num_materials];
    for (int i = 0; i < this->num_materials; i++)
    {
        temp_materials[i] = nullptr;
        this->materials_data[i]->MallocCuda(temp_materials[i]);
    }
    cudaMemcpy(temp.materials_data, temp_materials, sizeof(Material*) * this->num_materials, cudaMemcpyHostToDevice);
    delete[] temp_materials;
    // 3. upload meshes
    AllocateAndRegister(this->meshes_data, temp.meshes_data, this->num_meshes);
    MeshTriangle** temp_meshes = new MeshTriangle * [this->num_meshes];
    for (int i = 0; i < this->num_meshes; i++)
    {
        temp_meshes[i] = nullptr;
        this->meshes_data[i]->MallocCuda(temp_meshes[i]);
    }
    cudaMemcpy(temp.meshes_data, temp_meshes, sizeof(MeshTriangle*) * this->num_meshes, cudaMemcpyHostToDevice);
    delete[] temp_meshes;

    cudaMemcpy(device_ptr, &temp, sizeof(Scene), cudaMemcpyHostToDevice);
}
void Scene::FreeCuda() const
{
    // 1. free bvh
    this->bvh->FreeCuda();
    // 2. free material
    for (int i = 0; i < this->num_materials; i++)
    {
        this->materials_data[i]->FreeCuda();
    }
    FreeAndUnregister(this->materials_data);
    // 3. free meshes
    for (int i = 0; i < this->num_meshes; i++)
    {
        this->meshes_data[i]->FreeCuda();
    }
    FreeAndUnregister(this->meshes_data);

    FreeAndUnregister(this);
}