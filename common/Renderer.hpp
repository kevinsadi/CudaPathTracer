//
// Created by goksu on 2/25/20.
//
#pragma once
#include "Scene.hpp"
#include <glm/glm.hpp>

struct hit_payload
{
    float tNear;
    uint32_t index;
    glm::vec2 uv;
    Object* hit_obj;
};

class Renderer
{
public:
    int spp = 32;
    std::vector<glm::vec3> framebuffer;
    virtual void Render(const Scene& scene);

private:
};
