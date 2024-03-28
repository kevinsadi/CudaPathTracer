//
// Created by goksu on 2/25/20.
//
#pragma once
#include "Scene.hpp"

struct hit_payload
{
    float tNear;
    uint32_t index;
    Vector2f uv;
    Object* hit_obj;
};

class Renderer
{
public:
    int spp = 32;
    std::vector<Vector3f> framebuffer;
    void Render(const Scene& scene);

private:
};
