#include "global.hpp"
#include <random>

int RandSeed = 0;
static std::random_device dev;
std::mt19937 Rng(dev());
void SetRandomSeed(int seed) {
    RandSeed = seed;
    Rng.seed(seed);
}
int GetRandomSeed()
{
    return RandSeed;
}