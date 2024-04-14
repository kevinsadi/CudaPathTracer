//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BVH_H
#define RAYTRACING_BVH_H

#include <atomic>
#include <vector>
#include <memory>
#include <ctime>
#include "Triangle.hpp"
#include "Ray.hpp"
#include "Bounds3.hpp"
#include "Intersection.hpp"
#include <unordered_map>

struct BVHBuildNode;
// BVHAccel Forward Declarations
// struct BVHPrimitiveInfo;

// // BVHAccel Declarations
// inline int leafNodes, totalLeafNodes, totalPrimitives, interiorNodes;
class BVHAccel {

public:
    // BVHAccel Public Types
    enum class SplitMethod { NAIVE, SAH };

    // BVHAccel Public Methods
    BVHAccel(std::vector<Object*>& p, std::unordered_map<MeshTriangle*, BVHAccel*>& meshBvhMap, int maxPrimsInNode = 1, SplitMethod splitMethod = SplitMethod::NAIVE);
    FUNC_QUALIFIER inline Bounds3 WorldBound() const;
    ~BVHAccel();

    FUNC_QUALIFIER inline Intersection Intersect(const Ray& ray) const;
    FUNC_QUALIFIER inline Intersection getIntersection(BVHBuildNode* node, const Ray& ray)const;
    BVHBuildNode* root = nullptr;

    // BVHAccel Private Methods
    BVHBuildNode* recursiveBuild(std::vector<Object*>& objects, std::unordered_map<MeshTriangle*, BVHAccel*>& meshBvhMap);

    // BVHAccel Private Data
    const int maxPrimsInNode;
    const SplitMethod splitMethod;
    // std::vector<Object*> primitives;
    // Object** primitives = nullptr;
    // int num_primitives = 0;

    FUNC_QUALIFIER inline void getSample(RNG& rng, BVHBuildNode* node, float p, Intersection& pos, float& pdf);
    FUNC_QUALIFIER inline void Sample(RNG& rng, Intersection& pos, float& pdf);

    CUDA_PORTABLE(BVHAccel);
};

struct BVHBuildNode {
    static int Counter;
    int id = Counter++;
    Bounds3 bounds;
    BVHBuildNode* left;
    BVHBuildNode* right;
    MeshTriangle* mesh;
    BVHAccel* meshBvhRoot;
    Triangle* triangle;
    float area;
    BVHBuildNode* nextIfHit = nullptr; // blue
    BVHBuildNode* nextIfMiss = nullptr; // red

public:
    int splitAxis = 0, firstPrimOffset = 0, nPrimitives = 0;
    // BVHBuildNode Public Methods
    BVHBuildNode() {
        bounds = Bounds3();
        left = nullptr;right = nullptr;nextIfHit = nullptr;nextIfMiss = nullptr;
        // object = nullptr;
        meshBvhRoot = nullptr;mesh = nullptr;triangle = nullptr;
    }
    ~BVHBuildNode();

    CUDA_PORTABLE(BVHBuildNode);
};

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    Intersection isect;
    // const std::array<int, 3> dirIsNeg = {ray.direction.x < 0, ray.direction.y < 0,
    //                        ray.direction.z < 0};
    const int dirIsNeg[3] = { ray.direction.x < 0, ray.direction.y < 0,
                           ray.direction.z < 0 };
    // if (!node->bounds.IntersectP(ray, ray.direction_inv, dirIsNeg))
    //     return isect;
    // // leaf node
    // if (node->left == nullptr && node->right == nullptr)
    //     return node->object->getIntersection(ray);

    // Intersection hitLeft = getIntersection(node->left, ray);
    // Intersection hitRight = getIntersection(node->right, ray);

    // return hitLeft.distance < hitRight.distance ? hitLeft : hitRight;
    BVHBuildNode* current = node;
    while (current)
    {
        bool boundsHit = current->bounds.IntersectP(ray, ray.direction_inv, dirIsNeg);
        if (boundsHit)
        {
            // if leaf node, compute geometry-ray intersection, compare hit distance
            if (current->left == nullptr && current->right == nullptr)
            {
                Intersection hit = current->mesh ? current->meshBvhRoot->Intersect(ray)
                    : current->triangle->getIntersection(ray);
                if (hit.distance < isect.distance)
                    isect = hit;
            }
            current = current->nextIfHit;
        }
        else
        {
            current = current->nextIfMiss;
        }
    }
    return isect;
}

void BVHAccel::getSample(RNG& rng, BVHBuildNode* node, float p, Intersection& pos, float& pdf) {
    // if (node->left == nullptr || node->right == nullptr) {
    //     if (node->mesh) node->mesh->Sample(pos, pdf);
    //     else node->triangle->Sample(pos, pdf);
    //     pdf *= node->area;
    //     return;
    // }
    // if (p < node->left->area) getSample(node->left, p, pos, pdf);
    // else getSample(node->right, p - node->left->area, pos, pdf);
    BVHBuildNode* current = node;
    while (current)
    {
        // if leaf node
        if (current->left == nullptr && current->right == nullptr)
        {
            current->triangle->Sample(rng, pos, pdf);
            // pdf *= current->area;
            break;
        }

        if (p < current->left->area)
        {
            current = current->left;
        }
        else
        {
            p -= current->left->area;
            current = current->right;
        }
    }
}

void BVHAccel::Sample(RNG& rng, Intersection& pos, float& pdf) {
    float p = glm::sqrt(rng.sample1D()) * root->area;
    getSample(rng, root, p, pos, pdf);
    // pdf /= root->area;
    pdf = 1.0f / root->area;
}



#endif //RAYTRACING_BVH_H
