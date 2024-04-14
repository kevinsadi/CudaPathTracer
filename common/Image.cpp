#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Image.h"

Image::Image(int width, int height) {
    mWidth = width;
    mHeight = height;
}

Image::Image(const std::string& path) {
    int width, height, nrComponents;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
    
    if (!data) {
        std::cout << "Failed to load image: " + path << std::endl;
    }
    mData = new glm::vec3[width * height];
    memcpy(mData, data, width * height * sizeof(glm::vec3));

    mWidth = width;
    mHeight = height;

    if (data) {
        free(data);
    }
}