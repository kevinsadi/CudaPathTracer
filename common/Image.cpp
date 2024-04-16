#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Image.hpp"

Image::Image(int width, int height) {
    mWidth = width;
    mHeight = height;
    mData = new glm::vec3[width * height];
}

Image::Image(const std::string& path) {
    int width, height, nrComponents;
    float* data = stbi_loadf(path.c_str(), &width, &height, &nrComponents, 0);
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

glm::vec3 Image::linearSample(glm::vec2 uv) {
        const float Eps = FLT_MIN;
        uv = glm::fract(uv);

        float fx = uv.x * (mWidth - Eps) + .5f;
        float fy = uv.y * (mHeight - Eps) + .5f;

        int ix = glm::fract(fx) > .5f ? fx : fx - 1;
        if (ix < 0) {
            ix += mWidth;
        }

        int iy = glm::fract(fy) > .5f ? fy : fy - 1;
        if (iy < 0) {
            iy += mHeight;
        }

        int ux = ix + 1;
        if (ux >= mWidth) {
            ux -= mWidth;
        }

        int uy = iy + 1;
        if (uy >= mHeight) {
            uy -= mHeight;
        }

        float lx = glm::fract(fx + .5f);
        float ly = glm::fract(fy + .5f);

        glm::vec3 c1 = glm::mix(getTexel(ix, iy), getTexel(ux, iy), lx);
        glm::vec3 c2 = glm::mix(getTexel(ix, uy), getTexel(ux, uy), lx);
        return glm::mix(c1, c2, ly);
}

glm::vec3 Image::getTexel(int x, int y) {
    return mData[y * mWidth + x];
}