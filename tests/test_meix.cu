#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/meix.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"

#include <iostream>

const char* test_torus()
{
    const auto n_bolls = 1000;
    Solution<float3, n_bolls, Grid_solver> bolls;
    uniform_cuboid(-1.f, -1.f, -0.5f, 2.f, 2.f, 1.f, bolls);

    Meix meix("tests/torus.vtk");
    auto in = meix.test_inclusion(bolls);

    for (auto i = 0; i < n_bolls; i++) {
        auto dist_from_ring = sqrt(
            pow(1 - sqrt(pow(bolls.h_X[i].x, 2) + pow(bolls.h_X[i].y, 2)), 2) +
            pow(bolls.h_X[i].z, 2));
        if (abs(dist_from_ring - 0.5) < 0.01) continue;  // Tolerance for mesh

        MU_ASSERT("Inclusion test wrong", (dist_from_ring <= 0.5) == in[i]);
    }

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_torus);
    return NULL;
}

MU_RUN_SUITE(all_tests);
