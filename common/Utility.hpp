#include <vector>
#include <string>
#include <common/Vector.hpp>

class Utility
{
public:
    static void SavePPM(const std::string& path, const std::vector<Vector3f>& frameBuffer, int width, int height);
    static void UpdateProgress(float progress);
};