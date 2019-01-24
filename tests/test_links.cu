#include "../include/dtypes.cuh"
#include "../include/links.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


template<typename Pt>
__device__ Pt no_pw_int(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};
    return dF;
}


const char* test_square_of_four()
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


template<typename Pt>
__device__ void custom_force(const Pt* __restrict__ d_X, const int a,
    const int b, const float strength, Pt* d_dX)
{
    atomicAdd(&d_dX[a].w, -1);
    atomicAdd(&d_dX[b].w, 1);
}

const char* test_custom_force()
{
    Solution<float4, Tile_solver> points{2};
    Links links{1};

    auto forces = [&links](const float4* __restrict__ d_X, float4* d_dX) {
        return link_forces(links, d_X, d_dX);
    };

    auto custom_forces = [&links](const float4* __restrict__ d_X, float4* d_dX) {
        return link_forces<float4, custom_force>(links, d_X, d_dX);
    };

    // clang-format off
    points.h_X[0].x = 1;  points.h_X[0].y = 1;  points.h_X[0].z = 0; points.h_X[0].w = 1;
    points.h_X[1].x = 1;  points.h_X[1].y = -1; points.h_X[1].z = 0; points.h_X[1].w = -1;
    links.h_link[0].a = 0; links.h_link[0].b = 1;
    // clang-format on
    points.copy_to_device();
    links.copy_to_device();

    auto dt = 0.1;
    points.take_step<no_pw_int>(dt, forces);
    points.take_step<no_pw_int>(dt, custom_forces);

    points.copy_to_host();
    MU_ASSERT("Wrong x", isclose(points.h_X[0].x - points.h_X[1].x, 0));
    MU_ASSERT("Wrong y", isclose(points.h_X[0].y - points.h_X[1].y, 2 - 2 * dt * links.strength));
    MU_ASSERT("Wrong z", isclose(points.h_X[0].z - points.h_X[1].z, 0));
    MU_ASSERT("Wrong w", isclose(points.h_X[0].w - points.h_X[1].w, 2 - 2 * dt));

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_square_of_four);
    MU_RUN_TEST(test_custom_force);
    return NULL;
}

MU_RUN_SUITE(all_tests);
