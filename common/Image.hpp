#pragma once


#include <iostream>
#include <glm/glm.hpp>

class Image {
public:
    Image(int width, int height);
    Image(const std::string& path);
    ~Image();

    glm::vec3 getTexel(int x, int y);
    glm::vec3 linearSample(glm::vec2 uv);

private:
    int mWidth;
    int mHeight;
    glm::vec3* mData = nullptr;
};