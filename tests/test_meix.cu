#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/meix.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


const char* test_transformations()
{
    Meix meix("tests/torus.vtk");
    auto minimum = meix.get_minimum();
    auto maximum = meix.get_maximum();
    MU_ASSERT("Min wrong in x", isclose(minimum.x, -1.5));
    MU_ASSERT("Min wrong in y", isclose(minimum.y, -1.5));
    MU_ASSERT("Min wrong in z", isclose(minimum.z, -0.5));
    MU_ASSERT("Max wrong in x", isclose(maximum.x, 1.5));
    MU_ASSERT("Max wrong in y", isclose(maximum.y, 1.5));
    MU_ASSERT("Max wrong in z", isclose(maximum.z, 0.5));

    meix.translate(float3{1, 0, 0});
    minimum = meix.get_minimum();
    maximum = meix.get_maximum();
    MU_ASSERT("Translated min wrong in x", isclose(minimum.x, -1.5 + 1));
    MU_ASSERT("Translated min wrong in y", isclose(minimum.y, -1.5));
    MU_ASSERT("Translated min wrong in z", isclose(minimum.z, -0.5));
    MU_ASSERT("Translated max wrong in x", isclose(maximum.x, 1.5 + 1));
    MU_ASSERT("Translated max wrong in y", isclose(maximum.y, 1.5));
    MU_ASSERT("Translated max wrong in z", isclose(maximum.z, 0.5));
    meix.translate(float3{-1, 0, 0});

    meix.rotate(0, M_PI / 2, 0);
    minimum = meix.get_minimum();
    maximum = meix.get_maximum();
    MU_ASSERT("Rotated min wrong in x", isclose(minimum.x, -0.5));
    MU_ASSERT("Rotated min wrong in y", isclose(minimum.y, -1.5));
    MU_ASSERT("Rotated min wrong in z", isclose(minimum.z, -1.5));
    MU_ASSERT("Rotated max wrong in x", isclose(maximum.x, 0.5));
    MU_ASSERT("Rotated max wrong in y", isclose(maximum.y, 1.5));
    MU_ASSERT("Rotated max wrong in z", isclose(maximum.z, 1.5));
    meix.rotate(0, -M_PI / 2, 0);

    meix.rescale(2);
    minimum = meix.get_minimum();
    maximum = meix.get_maximum();
    MU_ASSERT("Scaled min wrong in x", isclose(minimum.x, -1.5 * 2));
    MU_ASSERT("Scaled min wrong in y", isclose(minimum.y, -1.5 * 2));
    MU_ASSERT("Scaled min wrong in z", isclose(minimum.z, -0.5 * 2));
    MU_ASSERT("Scaled max wrong in x", isclose(maximum.x, 1.5 * 2));
    MU_ASSERT("Scaled max wrong in y", isclose(maximum.y, 1.5 * 2));
    MU_ASSERT("Scaled max wrong in z", isclose(maximum.z, 0.5 * 2));
    meix.rescale(0.5);

    meix.grow_normally(0.1);
    minimum = meix.get_minimum();
    maximum = meix.get_maximum();
    MU_ASSERT("Grown min wrong in x", isclose(minimum.x, -1.5 - 0.1));
    MU_ASSERT("Grown min wrong in y", isclose(minimum.y, -1.5 - 0.1));
    MU_ASSERT("Grown min wrong in z", isclose(minimum.z, -0.5 - 0.1));
    MU_ASSERT("Grown max wrong in x", isclose(maximum.x, 1.5 + 0.1));
    MU_ASSERT("Grown max wrong in y", isclose(maximum.y, 1.5 + 0.1));
    MU_ASSERT("Grown max wrong in z", isclose(maximum.z, 0.5 + 0.1));

    return NULL;
}


const char* test_exclusion()
{
    const auto n_bolls = 1500;
    Solution<float3, n_bolls, Grid_solver> bolls;
    random_cuboid(0.25, float3{-1.5, -1.5, -0.5}, float3{1.5, 1.5, 0.5}, bolls);

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


const char* test_shape_comparison()
{
    Meix meix("tests/torus.vtk");
    Solution<float3, 987, Grid_solver> bolls;
    for (auto i = 0; i < 987; i++) {
        bolls.h_X[i].x = meix.vertices[i].x;
        bolls.h_X[i].y = meix.vertices[i].y;
        bolls.h_X[i].z = meix.vertices[i].z;
    }
    bolls.copy_to_device();

    MU_ASSERT("Shape comparison wrong",
        isclose(meix.shape_comparison_distance_meix_to_bolls(bolls), 0.0));

    meix.grow_normally(0.1);
    MU_ASSERT("Grown shape comparison wrong",
        isclose(meix.shape_comparison_distance_meix_to_bolls(bolls), 0.1));

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_transformations);
    MU_RUN_TEST(test_exclusion);
    MU_RUN_TEST(test_shape_comparison);
    return NULL;
}

MU_RUN_SUITE(all_tests);
