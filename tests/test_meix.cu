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
    uniform_cuboid(-1.5, -1.5, -0.5, 3, 3, 1, bolls);

    Meix meix("tests/torus.vtk");
    for (auto i = 0; i < n_bolls; i++) {
        auto dist_from_ring = sqrt(
            pow(1 - sqrt(pow(bolls.h_X[i].x, 2) + pow(bolls.h_X[i].y, 2)), 2) +
            pow(bolls.h_X[i].z, 2));
        if (abs(dist_from_ring - 0.5) < 0.01) continue;  // Tolerance for mesh

        auto out = meix.test_exclusion(bolls.h_X[i]);
        MU_ASSERT("Exclusion test wrong", (dist_from_ring >= 0.5) == out);
    }

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_torus);
    return NULL;
}

MU_RUN_SUITE(all_tests);
