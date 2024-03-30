#include "CudaPathTrace.h"
#include <common/CudaPortable.hpp>
#include <common/Ray.hpp>
#include <common/Material.hpp>
#include <common/Triangle.hpp>
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
    if (host == nullptr)
    {
        throw std::runtime_error("host pointer is nullptr");
    }
    cudaMalloc(&device, sizeof(T) * count);
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
CUDA_AUTO_ALLOCATION(Material);
CUDA_AUTO_ALLOCATION(Triangle);
// otherwise, we need to manually implement the MallocCuda and FreeCuda function

static std::unordered_map<const BVHBuildNode*, BVHBuildNode*> bvhMap; // key: host bvh node, value: device bvh node(not copied to device yet)
void ClearBVHMap()
{
    if (!bvhMap.empty())
    {
        for (auto& pair : bvhMap)
        {
            pair.second->left = nullptr;
            pair.second->right = nullptr;
            pair.second->meshBvhRoot = nullptr;
            delete pair.second;
        }
    }
    bvhMap.clear();
}
void UploadBVHToGPU()
{
    for (auto& pair : bvhMap)
    {
        BVHBuildNode* host_ptr = const_cast<BVHBuildNode*>(pair.first);
        BVHBuildNode* host_temp_ptr = pair.second;
        BVHBuildNode* device_ptr = (BVHBuildNode*)GetDeviceMemory((void*)host_ptr);

        // set nextIfHit and nextIfMiss
        if (host_ptr->nextIfHit)
        {
            host_temp_ptr->nextIfHit = (BVHBuildNode*)GetDeviceMemory((void*)host_ptr->nextIfHit);
        }
        else
        {
            host_temp_ptr->nextIfHit = nullptr;
        }
        if (host_ptr->nextIfMiss)
        {
            host_temp_ptr->nextIfMiss = (BVHBuildNode*)GetDeviceMemory((void*)host_ptr->nextIfMiss);
        }
        else
        {
            host_temp_ptr->nextIfMiss = nullptr;
        }

        // copy host_temp to device
        cudaMemcpy(device_ptr, host_temp_ptr, sizeof(BVHBuildNode), cudaMemcpyHostToDevice);
    }
}
// no need to unload, as the device memory is freed when the scene is freed

void BVHBuildNode::MallocCuda(BVHBuildNode*& device_ptr) const
{
    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    BVHBuildNode* temp = new BVHBuildNode();
    *temp = *this;
    // add to bvhMap
    if (bvhMap.find(this) == bvhMap.end())
    {
        bvhMap[this] = temp;
    }
    else
    {
        throw std::runtime_error("Object already exists in bvhMap: object being allocated is already uploaded to device memory");
    }

    if (this->mesh)
    {
        temp->mesh = (MeshTriangle*)GetDeviceMemory((void*)this->mesh);
        temp->meshBvhRoot = nullptr;
        this->meshBvhRoot->MallocCuda(temp->meshBvhRoot);
    }
    else if (this->triangle)
    {
        temp->triangle = (Triangle*)GetDeviceMemory((void*)this->triangle);
    }
    else
    {
        assert (this->left != nullptr && this->right != nullptr);
        temp->left = nullptr;
        this->left->MallocCuda(temp->left);
        temp->right = nullptr;
        this->right->MallocCuda(temp->right);
    }
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
    if (this->meshBvhRoot != nullptr)
    {
        this->meshBvhRoot->FreeCuda();
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
    temp.root = nullptr;
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

void MeshTriangle::MallocCuda(MeshTriangle*& device_ptr) const
{
    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    MeshTriangle temp = *this;
    assert(this->vertices != nullptr);
    assert(this->stCoordinates != nullptr);
    assert(this->vertexIndex != nullptr);
    assert(this->triangles != nullptr);
    // 2. upload indexed mesh data to device
    AllocateAndRegister(this->vertices, temp.vertices, temp.num_vertices);
    AllocateAndRegister(this->stCoordinates, temp.stCoordinates, temp.num_vertices);
    AllocateAndRegister(this->vertexIndex, temp.vertexIndex, temp.num_triangles * 3);
    cudaMemcpy(temp.vertices, this->vertices, sizeof(glm::vec3) * temp.num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(temp.stCoordinates, this->stCoordinates, sizeof(glm::vec2) * temp.num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(temp.vertexIndex, this->vertexIndex, sizeof(uint32_t) * temp.num_triangles * 3, cudaMemcpyHostToDevice);
    // 3. upload triangle soup to device
    AllocateAndRegister(this->triangles, temp.triangles, temp.num_triangles);
    UnregisterMemory((void*)this->triangles, (void*)temp.triangles); // trick, not use this often
    for (int i = 0; i < temp.num_triangles; i++)
    {
        // allocate triangle on continuous memory
        Triangle* triangle_ptr = temp.triangles + i;
        this->triangles[i].MallocCuda(triangle_ptr);
    }

    cudaMemcpy(device_ptr, &temp, sizeof(MeshTriangle), cudaMemcpyHostToDevice);
    temp.vertices = nullptr;
    temp.stCoordinates = nullptr;
    temp.vertexIndex = nullptr;
    // temp.bvh = nullptr;
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

    FreeAndUnregister(this);
}

void Scene::MallocCuda(Scene*& device_ptr) const
{
    // 0. clear bvh map
    ClearBVHMap();

    AllocateAndRegisterIfNullptr(this, device_ptr, 1);
    Scene temp = *this;
    assert(this->meshes_data != nullptr);
    assert(this->bvh != nullptr);
    // 2. upload meshes
    AllocateAndRegister(this->meshes_data, temp.meshes_data, this->num_meshes);
    MeshTriangle** temp_meshes = new MeshTriangle * [this->num_meshes];
    for (int i = 0; i < this->num_meshes; i++)
    {
        temp_meshes[i] = nullptr;
        this->meshes_data[i]->MallocCuda(temp_meshes[i]);
    }
    cudaMemcpy(temp.meshes_data, temp_meshes, sizeof(MeshTriangle*) * this->num_meshes, cudaMemcpyHostToDevice);
    delete[] temp_meshes;
    // 3. create scene bvh
    temp.bvh = nullptr;
    this->bvh->MallocCuda(temp.bvh);
    AllocateAndRegister(this->mesh_bvhs, temp.mesh_bvhs, this->num_meshes);
    BVHAccel** temp_mesh_bvhs = new BVHAccel * [this->num_meshes];
    for (int i = 0; i < this->num_meshes; i++)
    {
        temp_mesh_bvhs[i] = (BVHAccel*)GetDeviceMemory((void*)this->mesh_bvhs[i]);
    }
    cudaMemcpy(temp.mesh_bvhs, temp_mesh_bvhs, sizeof(BVHAccel*) * this->num_meshes, cudaMemcpyHostToDevice);
    delete[] temp_mesh_bvhs;
    // 4. upload all bvh
    UploadBVHToGPU();
    ClearBVHMap();

    cudaMemcpy(device_ptr, &temp, sizeof(Scene), cudaMemcpyHostToDevice);
}
void Scene::FreeCuda() const
{
    // 2. free meshes
    for (int i = 0; i < this->num_meshes; i++)
    {
        this->meshes_data[i]->FreeCuda();
    }
    FreeAndUnregister(this->meshes_data);
    // 3. free bvh
    this->bvh->FreeCuda();
    FreeAndUnregister(this->mesh_bvhs);

    FreeAndUnregister(this);
}