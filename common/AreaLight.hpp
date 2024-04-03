// //
// // Created by Göksu Güvendiren on 2019-05-14.
// //

// #pragma once

// #include "MathUtils.hpp"
// #include "Light.hpp"

// class AreaLight : public Light
// {
// public:
//     AreaLight(const glm::vec3 &p, const glm::vec3 &i) : Light(p, i)
//     {
//         normal = glm::vec3(0, -1, 0);
//         u = glm::vec3(1, 0, 0);
//         v = glm::vec3(0, 0, 1);
//         length = 100;
//     }

//     glm::vec3 SamplePoint() const
//     {
//         auto random_u = get_random_float();
//         auto random_v = get_random_float();
//         return position + random_u * u + random_v * v;
//     }

//     float length;
//     glm::vec3 normal;
//     glm::vec3 u;
//     glm::vec3 v;
// };
