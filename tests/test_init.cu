#include <math.h>
#include <thrust/transform_reduce.h>
#include <array>
#include <functional>

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

    relaxed_cuboid(r_mean, float3{0}, float3{9, 9, 9}, bolls);
    pos_before = store(bolls);
    bolls.take_step<relu_force>(dt);
    bolls.copy_to_host();
    pos_after = store(bolls);
    diff = mean_difference<n_max>(pos_before, pos_after, *bolls.h_n);
    MU_ASSERT("Cuboid not relaxed", MU_ISCLOSE(diff, 0));

    return NULL;
}


template<typename Pt>
float3 compwise_min(const Pt& a, const Pt& b)
{
    return float3{min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

const char* test_cuboid_dimensions()
{
    const auto r_mean = 0.8;
    const auto n_max = 5000;
    Solution<float3, n_max, Grid_solver> bolls;

    relaxed_cuboid(r_mean, float3{0}, float3{9, 9, 9}, bolls);

    auto mins = thrust::reduce(bolls.h_X, bolls.h_X + *bolls.h_n, bolls.h_X[0],
        compwise_min<float3>);
    MU_ASSERT("Cuboid too small in x", mins.x < 0);
    MU_ASSERT("Cuboid too small in y", mins.y < 0);
    MU_ASSERT("Cuboid too small in z", mins.z < 0);

    MU_ASSERT("Cuboid too large in x", mins.x > -r_mean);
    MU_ASSERT("Cuboid too large in y", mins.y > -r_mean);
    MU_ASSERT("Cuboid too large in z", mins.z > -r_mean);

    return NULL;
}


const char* all_tests()
{
    // MU_RUN_TEST(test_relaxation);
    MU_RUN_TEST(test_cuboid_dimensions);
    return NULL;
}

MU_RUN_SUITE(all_tests);
