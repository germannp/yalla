#include <array>
#include "math.h"

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


template<typename Pt, int n_max, template<typename, int> class Solver>
std::array<Pt, n_max> store(Solution<Pt, n_max, Solver>& bolls)
{
    std::array<float3, n_max> position;
    for (auto i = 0; i < n_max; i++) {
        position[i] = bolls.h_X[i];
    }
    return position;
}

template<int n_max>
float mean_difference(
    std::array<float3, n_max>& a, std::array<float3, n_max>& b, int n = n_max)
{
    auto total_diff = 0.f;
    for (int i = 0; i < n; i++) {
        auto diff = pow(a[i].x - b[i].x, 2) + pow(a[i].y - b[i].y, 2) +
                     pow(a[i].z - b[i].z, 2);
        total_diff += sqrt(diff);
    }
    return total_diff / n_max;
}

const char* test_relaxation()
{
    const auto r_mean = 0.8;
    const auto dt = 0.1;
    const auto n_max = 5000;
    Solution<float3, n_max, Grid_solver> bolls;

    relaxed_sphere(r_mean, bolls);
    auto pos_before = store(bolls);
    bolls.take_step<relu_force>(dt);
    bolls.copy_to_host();
    auto pos_after = store(bolls);
    auto diff = mean_difference<n_max>(pos_before, pos_after, *bolls.h_n);
    MU_ASSERT("Sphere not relaxed", MU_ISCLOSE(diff, 0));

    relaxed_cuboid(r_mean, float3{0}, float3{7, 7, 7}, bolls);
    pos_before = store(bolls);
    bolls.take_step<relu_force>(dt);
    bolls.copy_to_host();
    pos_after = store(bolls);
    diff = mean_difference<n_max>(pos_before, pos_after, *bolls.h_n);
    MU_ASSERT("Cuboid not relaxed", MU_ISCLOSE(diff, 0));

    return NULL;
}



const char* all_tests()
{
    MU_RUN_TEST(test_relaxation);
    MU_RUN_TEST(test_cuboid_dimensions);
    return NULL;
}

MU_RUN_SUITE(all_tests);
