#pragma once


#include <iostream>
#include <glm/glm.hpp>

class Image {
public:
    Image(int width, int height);
    Image(const std::string& path);
    ~Image();

private:
    int mWidth;
    int mHeight;
    glm::vec3* mData = nullptr;
};