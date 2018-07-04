#include <math.h>
#include <thrust/transform_reduce.h>
#include <array>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


template<int n_max>
std::array<float3, n_max> store(Solution<float3, Grid_solver>& points)
{
    std::array<float3, n_max> position;
    for (auto i = 0; i < n_max; i++) { position[i] = points.h_X[i]; }
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

template<typename Pt, template<typename> class Solver>
float mean_dist_to_nbs(Solution<Pt, Solver>& points, float cut_off = 1)
{
    auto rnd_cell = static_cast<int>(rand() / (RAND_MAX + 1.) * *points.h_n);
    auto tot_dist = 0.f;
    auto n_nbs = 0;
    for (auto i = 0; i < *points.h_n; i++) {
        auto dist = sqrt(pow(points.h_X[rnd_cell].x - points.h_X[i].x, 2) +
                         pow(points.h_X[rnd_cell].y - points.h_X[i].y, 2) +
                         pow(points.h_X[rnd_cell].z - points.h_X[i].z, 2));
        if (dist < cut_off and i != rnd_cell) {
            tot_dist += dist;
            n_nbs++;
        }
    }
    return tot_dist / n_nbs;
}

const char* test_relaxation()
{
    const auto r_mean = 0.8;
    const auto dt = 0.1;
    const auto n_max = 5000;
    Solution<float3, Grid_solver> points{n_max};

    relaxed_sphere(r_mean, points);
    auto pos_before = store<n_max>(points);
    points.take_step<relu_force>(dt);
    points.copy_to_host();
    auto pos_after = store<n_max>(points);
    auto diff = mean_difference<n_max>(pos_before, pos_after, *points.h_n);
    MU_ASSERT("Sphere not relaxed", diff < 5e-4);
    auto mean_dist = mean_dist_to_nbs(points);
    MU_ASSERT("Sphere mean dist to neighbours wrong",
        r_mean - 0.05 < mean_dist and mean_dist < r_mean + 0.05);

    relaxed_cuboid(r_mean, float3{0}, float3{9, 9, 9}, points);
    pos_before = store<n_max>(points);
    points.take_step<relu_force>(dt);
    points.copy_to_host();
    pos_after = store<n_max>(points);
    diff = mean_difference<n_max>(pos_before, pos_after, *points.h_n);
    MU_ASSERT("Cuboid not relaxed", diff < 5e-4);
    mean_dist = mean_dist_to_nbs(points);
    MU_ASSERT("Cuboid mean dist to neighbours wrong",
        r_mean - 0.05 < mean_dist and mean_dist < r_mean + 0.05);

    return NULL;
}


template<typename Pt>
float3 compwise_min(const Pt& a, const Pt& b)
{
    return float3{min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

template<typename Pt>
float3 compwise_max(const Pt& a, const Pt& b)
{
    return float3{max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}

const char* test_cuboid_dimensions()
{
    const auto r_mean = 0.8;
    const auto n_max = 5000;
    Solution<float3, Grid_solver> points{n_max};

    relaxed_cuboid(r_mean, float3{0}, float3{9, 9, 9}, points);
    auto mins = thrust::reduce(points.h_X, points.h_X + *points.h_n,
        points.h_X[0], compwise_min<float3>);
    MU_ASSERT("Cuboid too small in x", mins.x < 0);
    MU_ASSERT("Cuboid too small in y", mins.y < 0);
    MU_ASSERT("Cuboid too small in z", mins.z < 0);
    MU_ASSERT("Cuboid too large in x", mins.x > -r_mean * 2);
    MU_ASSERT("Cuboid too large in y", mins.y > -r_mean * 2);
    MU_ASSERT("Cuboid too large in z", mins.z > -r_mean * 2);
    auto mean_dist = mean_dist_to_nbs(points);
    MU_ASSERT("2nd cuboid mean dist to neighbours wrong",
        r_mean - 0.05 < mean_dist and mean_dist < r_mean + 0.05);

    relaxed_cuboid(r_mean / 2, float3{0}, float3{4, 4, 4}, points);
    mins = thrust::reduce(points.h_X, points.h_X + *points.h_n, points.h_X[0],
        compwise_min<float3>);
    MU_ASSERT("Scaled cuboid too small in x", mins.x < 0);
    MU_ASSERT("Scaled cuboid too small in y", mins.y < 0);
    MU_ASSERT("Scaled cuboid too small in z", mins.z < 0);
    MU_ASSERT("Scaled cuboid too large in x", mins.x > -r_mean);
    MU_ASSERT("Scaled cuboid too large in y", mins.y > -r_mean);
    MU_ASSERT("Scaled cuboid too large in z", mins.z > -r_mean);
    auto maxs = thrust::reduce(points.h_X, points.h_X + *points.h_n,
        points.h_X[0], compwise_max<float3>);
    MU_ASSERT("Scaled cuboid too small in x", maxs.x > 4);
    MU_ASSERT("Scaled cuboid too small in y", maxs.y > 4);
    MU_ASSERT("Scaled cuboid too small in z", maxs.z > 4);
    MU_ASSERT("Scaled cuboid too large in x", maxs.x < 4 + r_mean);
    MU_ASSERT("Scaled cuboid too large in y", maxs.y < 4 + r_mean);
    MU_ASSERT("Scaled cuboid too large in z", maxs.z < 4 + r_mean);
    mean_dist = mean_dist_to_nbs(points, 0.5);
    MU_ASSERT("Scaled uboid mean dist to neighbours wrong",
        r_mean / 2 - 0.05 < mean_dist and mean_dist < r_mean / 2 + 0.05);

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_relaxation);
    MU_RUN_TEST(test_cuboid_dimensions);
    return NULL;
}

MU_RUN_SUITE(all_tests);
