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
    FUNC_QUALIFIER Bounds3 WorldBound() const;
    ~BVHAccel();

    FUNC_QUALIFIER Intersection Intersect(const Ray& ray) const;
    FUNC_QUALIFIER Intersection getIntersection(BVHBuildNode* node, const Ray& ray)const;
    BVHBuildNode* root;

    // BVHAccel Private Methods
    BVHBuildNode* recursiveBuild(std::vector<Object*>objects);

    // BVHAccel Private Data
    const int maxPrimsInNode;
    const SplitMethod splitMethod;
    // std::vector<Object*> primitives;
    // Object** primitives = nullptr;
    // int num_primitives = 0;

    FUNC_QUALIFIER void getSample(BVHBuildNode* node, float p, Intersection& pos, float& pdf);
    FUNC_QUALIFIER void Sample(Intersection& pos, float& pdf);

    CUDA_PORTABLE(BVHAccel);
};

struct BVHBuildNode {
    Bounds3 bounds;
    BVHBuildNode* left;
    BVHBuildNode* right;
    Object* object;
    float area;

public:
    int splitAxis = 0, firstPrimOffset = 0, nPrimitives = 0;
    // BVHBuildNode Public Methods
    BVHBuildNode() {
        bounds = Bounds3();
        left = nullptr;right = nullptr;
        object = nullptr;
    }

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
    const int dirIsNeg[3] = {ray.direction.x < 0, ray.direction.y < 0,
                           ray.direction.z < 0};
    if (!node->bounds.IntersectP(ray, ray.direction_inv, dirIsNeg))
        return isect;
    // leaf node
    if (node->left == nullptr && node->right == nullptr)
        return node->object->getIntersection(ray);

    Intersection hitLeft = getIntersection(node->left, ray);
    Intersection hitRight = getIntersection(node->right, ray);

    return hitLeft.distance < hitRight.distance ? hitLeft : hitRight;
}


void BVHAccel::getSample(BVHBuildNode* node, float p, Intersection &pos, float &pdf){
    if(node->left == nullptr || node->right == nullptr){
        node->object->Sample(pos, pdf);
        pdf *= node->area;
        return;
    }
    if(p < node->left->area) getSample(node->left, p, pos, pdf);
    else getSample(node->right, p - node->left->area, pos, pdf);
}

void BVHAccel::Sample(Intersection &pos, float &pdf){
    float p = glm::sqrt(get_random_float()) * root->area;
    getSample(root, p, pos, pdf);
    pdf /= root->area;
}



#endif //RAYTRACING_BVH_H
