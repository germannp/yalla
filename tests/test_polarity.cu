#include <random>

#include "../include/dtypes.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


const char* test_transformations()
{
    Polarity pol{static_cast<float>(acos(2. * rand() / (RAND_MAX + 1.) - 1)),
        static_cast<float>(rand() / (RAND_MAX + 1.) * 2 * M_PI)};
    auto inverse = pt_to_pol(pol_to_float3(pol));
    MU_ASSERT("Inverse wrong in theta", isclose(pol.theta, inverse.theta));
    MU_ASSERT("Inverse wrong in theta", isclose(pol.phi, inverse.phi));
    return NULL;
}


const char* test_polarization_force()
{
    Po_cell i{0.601, 0.305, 0.320, 0.209, 0.295};
    Po_cell j{0.762, 0.403, 0.121, 0.340, 0.431};

    auto dF = bidirectional_polarization_force(i, j);

    MU_ASSERT("Polarization force wrong in x", isclose(dF.x, 0));
    MU_ASSERT("Polarization force wrong in y", isclose(dF.y, 0));
    MU_ASSERT("Polarization force wrong in z", isclose(dF.z, 0));
    MU_ASSERT("Polarization force wrong in theta", isclose(dF.theta, 0.126));
    MU_ASSERT("Polarization force wrong in phi", isclose(dF.phi, 0.215));

    return NULL;
}


__device__ Po_cell bidirectional_polarization_force(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j or i == 1) return dF;

    dF += bidirectional_polarization_force(Xi, Xi - r);
    return dF;
}

const char* test_polarization()
{
    Solution<Po_cell, Tile_solver> points{2};

    // Turn in theta and phi, close to z-axis to test transformation
    Polarity p_i{M_PI / 2 + M_PI / 4 + 0.01, 0.5};
    Polarity p_f{M_PI / 2 + M_PI / 4 + 0.01, M_PI};
    auto arc_if = acosf(pol_dot_product(p_i, p_f));

    points.h_X[0].theta = p_i.theta;
    points.h_X[0].phi = p_i.phi;
    points.h_X[1].theta = p_f.theta;
    points.h_X[1].phi = p_f.phi;
    points.copy_to_device();

    for (auto i = 0; i < 5000; i++) {
        points.copy_to_host();
        points.take_step<bidirectional_polarization_force>(0.01);
        auto arc_i0 = acosf(pol_dot_product(p_i, points.h_X[0]));
        auto arc_0f = acosf(pol_dot_product(points.h_X[0], p_f));
        MU_ASSERT(
            "Polarity off great circle", isclose(arc_i0 + arc_0f, arc_if));
    }

    auto prod = pol_dot_product(points.h_X[0], points.h_X[1]);
    MU_ASSERT("Polarities not aligned", isclose(fabs(prod), 1));

    return NULL;
}


const char* test_bending_force()
{
    Po_cell i{0.935, 0.675, 0.649, 0.793, 0.073};
    Po_cell j{0.566, 0.809, 0.533, 0.297, 0.658};

    auto r = i - j;
    auto dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
    auto dF = bending_force(i, r, dist);

    MU_ASSERT("Bending force wrong in x", isclose(dF.x, 0.214));
    MU_ASSERT("Bending force wrong in y", isclose(dF.y, -0.971));
    MU_ASSERT("Bending force wrong in z", isclose(dF.z, -1.802));
    MU_ASSERT("Bending force wrong in theta", isclose(dF.theta, -0.339));
    MU_ASSERT("Bending force wrong in phi", isclose(dF.phi, 0.453));

    return NULL;
}


__device__ Po_cell bending_force(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > 1) return dF;

    auto F = 2 * (0.6 - dist) * (1 - dist) + powf(1 - dist, 2);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += bending_force(Xi, r, dist) * 0.2;
    return dF;
}

const char* test_line_of_four()
{
    Solution<Po_cell, Tile_solver> points{4};

    for (auto i = 0; i < 4; i++) {
        points.h_X[i].x = 0.733333 * cosf((i - 0.5) * M_PI / 3);
        points.h_X[i].y = 0.733333 * sinf((i - 0.5) * M_PI / 3);
        points.h_X[i].z = 0;
        points.h_X[i].theta = M_PI / 2;
        points.h_X[i].phi = (i - 0.5) * M_PI / 3;
    }
    points.copy_to_device();
    auto com_i = center_of_mass(points);
    for (auto i = 0; i < 500; i++) { points.take_step<bending_force>(0.5); }

    points.copy_to_host();
    for (auto i = 1; i < 4; i++) {
        auto prod = pol_dot_product(points.h_X[0], points.h_X[i]);
        MU_ASSERT("Epithelial polarity not aligned", isclose(prod, 1));
    }

    float3 r_01{points.h_X[1].x - points.h_X[0].x,
        points.h_X[1].y - points.h_X[0].y, points.h_X[1].z - points.h_X[0].z};
    float3 r_12{points.h_X[2].x - points.h_X[1].x,
        points.h_X[2].y - points.h_X[1].y, points.h_X[2].z - points.h_X[1].z};
    float3 r_23{points.h_X[3].x - points.h_X[2].x,
        points.h_X[3].y - points.h_X[2].y, points.h_X[3].z - points.h_X[2].z};
    MU_ASSERT("Cells not on line in x", isclose(r_01.x, r_12.x));
    MU_ASSERT("Cells not on line in x", isclose(r_12.x, r_23.x));
    MU_ASSERT("Cells not on line in y", isclose(r_01.y, r_12.y));
    MU_ASSERT("Cells not on line in y", isclose(r_12.y, r_23.y));
    MU_ASSERT("Cells not on line in z", isclose(r_01.z, r_12.z));
    MU_ASSERT("Cells not on line in z", isclose(r_12.z, r_23.z));

    auto com_f = center_of_mass(points);
    MU_ASSERT("Momentum in line in x", isclose(com_i.x, com_f.x));
    MU_ASSERT("Momentum in line in y", isclose(com_i.y, com_f.y));
    MU_ASSERT("Momentum in line in z", isclose(com_i.z, com_f.z));

    return NULL;
}


const char* test_orthonormal()
{
    float3 r{static_cast<float>(rand() / (RAND_MAX + 1.)),
        static_cast<float>(rand() / (RAND_MAX + 1.)),
        static_cast<float>(rand() / (RAND_MAX + 1.))};
    float3 p{static_cast<float>(rand() / (RAND_MAX + 1.)),
        static_cast<float>(rand() / (RAND_MAX + 1.)),
        static_cast<float>(rand() / (RAND_MAX + 1.))};
    p /= sqrt(dot_product(p, p));

    auto n = orthonormal(r, p);
    MU_ASSERT("Not orthogonal", isclose(dot_product(p, n), 0));
    MU_ASSERT("Not normal", isclose(dot_product(n, n), 1));

    return NULL;
}


const char* test_migration_force()
{
    Po_cell Xi{0}, Xj{0};
    Xi.theta = M_PI / 2;
    Xj.x = 1;
    Xj.y = 1e-3;

    auto Fi = migration_force(Xi, Xi - Xj, 1);
    MU_ASSERT("Migration force wrong in x", isclose(Fi.x, 0.6));
    MU_ASSERT("Migration force wrong in y", isclose(Fi.y, -0.8));
    MU_ASSERT("Migration force wrong in z", abs(Fi.z) < 5e-5);

    auto Fj = migration_force(Xj, Xj - Xi, 1);
    MU_ASSERT("Migration forces not inverse in x", isclose(Fi.x, -Fj.x));
    MU_ASSERT("Migration forces not inverse in y", isclose(Fi.y, -Fj.y));
    MU_ASSERT("Migration forces not inverse in z", isclose(Fi.z, -Fj.z));

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_transformations);
    MU_RUN_TEST(test_polarization_force);
    MU_RUN_TEST(test_polarization);
    MU_RUN_TEST(test_bending_force);
    MU_RUN_TEST(test_line_of_four);
    MU_RUN_TEST(test_orthonormal);
    MU_RUN_TEST(test_migration_force);
    return NULL;
}

MU_RUN_SUITE(all_tests);
