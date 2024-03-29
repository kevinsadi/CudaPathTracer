#include "Utility.hpp"
#include <common/global.hpp>

void Utility::SavePPM(const std::string& path, const std::vector<Vector3f>& frameBuffer, int width, int height)
{
    // save the final render to file
    FILE* fp = fopen(path.c_str(), "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (auto i = 0; i < height * width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, frameBuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, frameBuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, frameBuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);
}