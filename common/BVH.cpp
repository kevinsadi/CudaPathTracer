#include <algorithm>
#include <cassert>
#include "BVH.hpp"
#include "MathUtils.hpp"
#include <vector>
#include <unordered_map>

void preorder(BVHBuildNode* root, std::vector<BVHBuildNode*>& nodes) {
    if (root == NULL) {
        return;
    }
    nodes.emplace_back(root);
    preorder(root->left, nodes);
    preorder(root->right, nodes);
}
std::vector<BVHBuildNode*> preorderTraversal(BVHBuildNode* root) {
    std::vector<BVHBuildNode*>nodes;
    preorder(root, nodes);
    return nodes;
}

void SetNextIfMiss(BVHBuildNode* node, BVHBuildNode* val) {
    if (node->left == nullptr && node->right == nullptr) {
        return;
    }
    else if (node->left && node->right)
    {
        node->left->nextIfMiss = node->right;
        node->right->nextIfMiss = val;
        SetNextIfMiss(node->left, node->left->nextIfMiss);
        SetNextIfMiss(node->right, node->right->nextIfMiss);
    }
    else
    {
        throw std::runtime_error("Error: node has only one child");
    }
}

BVHAccel::BVHAccel(std::vector<Object*>& p, std::unordered_map<MeshTriangle*, BVHAccel*>& meshBvhMap, int maxPrimsInNode,
    SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod)
    //   ,primitives(std::move(p))
{
    // num_primitives = p.size();
    // if (num_primitives == 0)
    //     return;
    if (p.size() == 0)
        return;
    time_t start, stop;
    time(&start);

    root = recursiveBuild(p, meshBvhMap);
    // (1) use pre-order traversal to set the nextIfHit
    auto preOrderNodes = preorderTraversal(root);
    for (int i = 0; i < preOrderNodes.size(); i++) {
        if (i == preOrderNodes.size() - 1) {
            preOrderNodes[i]->nextIfHit = nullptr;
        }
        else {
            preOrderNodes[i]->nextIfHit = preOrderNodes[i + 1];
        }
    }
    // (2) use pre-order traversal to set nextIfMiss
    SetNextIfMiss(root, nullptr);
    // primitives = new Object*[num_primitives];
    // memcpy(primitives, p.data(), num_primitives * sizeof(Object*));

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    // printf(
    //     "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
    //     hrs, mins, secs);
}

int BVHBuildNode::Counter = 0;

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*>& objects, std::unordered_map<MeshTriangle*, BVHAccel*>& meshBvhMap)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        MeshTriangle* mesh = dynamic_cast<MeshTriangle*>(objects[0]);
        Triangle* triangle = dynamic_cast<Triangle*>(objects[0]);
        if (mesh)
        {
            node->mesh = mesh;
            std::vector<Object*> triangles(mesh->num_triangles);
            for (int i = 0; i < mesh->num_triangles; i++)
            {
                triangles[i] = mesh->triangles + i;
            }
            node->meshBvhRoot = new BVHAccel(triangles, meshBvhMap, 1, SplitMethod::NAIVE);
            meshBvhMap[mesh] = node->meshBvhRoot;
        }
        else
        {
            node->triangle = triangle;
        }
        node->left = nullptr;
        node->right = nullptr;
        node->area = objects[0]->area;
        return node;
    }
    else if (objects.size() == 2) {
        std::vector<Object*> leftShapes{ objects[0] };
        std::vector<Object*> rightShapes{ objects[1] };
        node->left = recursiveBuild(leftShapes, meshBvhMap);
        node->right = recursiveBuild(rightShapes, meshBvhMap);

        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
        return node;
    }
    else {
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
            Union(centroidBounds, objects[i]->getBounds().Centroid());
        int dim = centroidBounds.maxExtent();
        switch (dim) {
        case 0:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                    f2->getBounds().Centroid().x;
                });
            break;
        case 1:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                    f2->getBounds().Centroid().y;
                });
            break;
        case 2:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                    f2->getBounds().Centroid().z;
                });
            break;
        }

        auto beginning = objects.begin();
        auto middling = objects.begin() + (objects.size() / 2);
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes, meshBvhMap);
        node->right = recursiveBuild(rightshapes, meshBvhMap);

        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
    }

    return node;
}

BVHAccel::~BVHAccel() {
    // if (primitives)
    // {
    //     delete[] primitives;
    // }
    if (root)
    {
        delete root;
    }
}

BVHBuildNode::~BVHBuildNode() {
    if (left)
        delete left;
    if (right)
        delete right;
    if (meshBvhRoot)
        delete meshBvhRoot;
}