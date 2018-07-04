#include "../include/dtypes.cuh"
#include "../include/links.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


__device__ float3 no_pw_int(float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    return dF;
}


const char* square_of_four()
{
    Solution<float3, Tile_solver> points{4};
    Links links{4};
    auto forces = [&links](const float3* __restrict__ d_X, float3* d_dX) {
        return link_forces(links, d_X, d_dX);
    };

    // clang-format off
    points.h_X[0].x = 1;  points.h_X[0].y = 1;  points.h_X[0].z = 0;
    points.h_X[1].x = 1;  points.h_X[1].y = -1; points.h_X[1].z = 0;
    points.h_X[2].x = -1; points.h_X[2].y = -1; points.h_X[2].z = 0;
    points.h_X[3].x = -1; points.h_X[3].y = 1;  points.h_X[3].z = 0;
    points.copy_to_device();
    links.h_link[0].a = 0; links.h_link[0].b = 1;
    links.h_link[1].a = 1; links.h_link[1].b = 2;
    links.h_link[2].a = 2; links.h_link[2].b = 3;
    links.h_link[3].a = 3; links.h_link[3].b = 0;
    // clang-format on
    links.copy_to_device();

    auto com_i = center_of_mass(points);
    for (auto i = 0; i < 500; i++) { points.take_step<no_pw_int>(0.1, forces); }

    points.copy_to_host();
    auto com_f = center_of_mass(points);
    MU_ASSERT("Momentum in square", isclose(com_i.x, com_f.x));
    MU_ASSERT("Momentum in square", isclose(com_i.y, com_f.y));
    MU_ASSERT("Momentum in square", isclose(com_i.z, com_f.z));

    MU_ASSERT("Not close in x", isclose(points.h_X[0].x, points.h_X[1].x));
    MU_ASSERT("Not close in y", isclose(points.h_X[1].y, points.h_X[2].y));
    MU_ASSERT("Not close in z", isclose(points.h_X[2].z, points.h_X[3].z));

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(square_of_four);
    return NULL;
}

MU_RUN_SUITE(all_tests);
