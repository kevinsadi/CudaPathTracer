//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BVH_H
#define RAYTRACING_BVH_H

#include <atomic>
#include <vector>
#include <memory>
#include <ctime>
#include "Object.hpp"
#include "Ray.hpp"
#include "Bounds3.hpp"
#include "Intersection.hpp"
#include "Vector.hpp"

struct BVHBuildNode;
// BVHAccel Forward Declarations
// struct BVHPrimitiveInfo;

// BVHAccel Declarations
inline int leafNodes, totalLeafNodes, totalPrimitives, interiorNodes;
class BVHAccel {

public:
    // BVHAccel Public Types
    enum class SplitMethod { NAIVE, SAH };

    // BVHAccel Public Methods
    BVHAccel(std::vector<Object*> p, int maxPrimsInNode = 1, SplitMethod splitMethod = SplitMethod::NAIVE);
    __host__ __device__ Bounds3 WorldBound() const;
    ~BVHAccel();

    __host__ __device__ Intersection Intersect(const Ray &ray) const;
    __host__ __device__ Intersection getIntersection(BVHBuildNode* node, const Ray& ray)const;
    __host__ __device__ bool IntersectP(const Ray &ray) const;
    BVHBuildNode* root;

    // BVHAccel Private Methods
    BVHBuildNode* recursiveBuild(std::vector<Object*>objects);

    // BVHAccel Private Data
    const int maxPrimsInNode;
    const SplitMethod splitMethod;
    // std::vector<Object*> primitives;
    // Object** primitives = nullptr;
    // int num_primitives = 0;

    __host__ __device__ void getSample(BVHBuildNode* node, float p, Intersection &pos, float &pdf);
    __host__ __device__ void Sample(Intersection &pos, float &pdf);

    CUDA_PORTABLE(BVHAccel);
};

struct BVHBuildNode {
    Bounds3 bounds;
    BVHBuildNode *left;
    BVHBuildNode *right;
    Object* object;
    float area;

public:
    int splitAxis=0, firstPrimOffset=0, nPrimitives=0;
    // BVHBuildNode Public Methods
    BVHBuildNode(){
        bounds = Bounds3();
        left = nullptr;right = nullptr;
        object = nullptr;
    }

    CUDA_PORTABLE(BVHBuildNode);
};




#endif //RAYTRACING_BVH_H
