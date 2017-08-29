#include "../include/dtypes.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"


const char* test_pcp_force()
{
    Po_cell i{0.601, 0.305, 0.320, 0.209, 0.295};
    Po_cell j{0.762, 0.403, 0.121, 0.340, 0.431};

    auto dF = pcp_force(i, j);

    MU_ASSERT("PCP force wrong in x", MU_ISCLOSE(dF.x, 0));
    MU_ASSERT("PCP force wrong in y", MU_ISCLOSE(dF.y, 0));
    MU_ASSERT("PCP force wrong in z", MU_ISCLOSE(dF.z, 0));
    MU_ASSERT("PCP force wrong in theta", MU_ISCLOSE(dF.theta, 0.126));
    MU_ASSERT("PCP force wrong in phi", MU_ISCLOSE(dF.phi, 0.215));

    return NULL;
}


__device__ Po_cell pcp_force(Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j or i == 1) return dF;

    dF += pcp_force(Xi, Xi - r);
    return dF;
}

const char* test_pcp()
{
    Solution<Po_cell, 2, Tile_solver> bolls;

    // Turn in theta and phi, close to z-axis to test transformation
    Polarity p_i{M_PI / 2 + M_PI / 4 + 0.01, 0.5};
    Polarity p_f{M_PI / 2 + M_PI / 4 + 0.01, M_PI};
    auto arc_if = acosf(pol_scalar_product(p_i, p_f));

    bolls.h_X[0].theta = p_i.theta;
    bolls.h_X[0].phi = p_i.phi;
    bolls.h_X[1].theta = p_f.theta;
    bolls.h_X[1].phi = p_f.phi;
    bolls.copy_to_device();

    for (auto i = 0; i < 5000; i++) {
        bolls.copy_to_host();
        bolls.take_step<pcp_force>(0.01);
        auto arc_i0 = acosf(pol_scalar_product(p_i, bolls.h_X[0]));
        auto arc_0f = acosf(pol_scalar_product(bolls.h_X[0], p_f));
        MU_ASSERT("PCP off great circle", MU_ISCLOSE(arc_i0 + arc_0f, arc_if));
    }

    auto prod = pol_scalar_product(bolls.h_X[0], bolls.h_X[1]);
    MU_ASSERT("PCP not aligned", MU_ISCLOSE(fabs(prod), 1));

    return NULL;
}


const char* test_rigidity_force()
{
    Po_cell i{0.935, 0.675, 0.649, 0.793, 0.073};
    Po_cell j{0.566, 0.809, 0.533, 0.297, 0.658};

    auto r = i - j;
    auto dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
    auto dF = rigidity_force(i, r, dist);

    MU_ASSERT("Rigidity force wrong in x", MU_ISCLOSE(dF.x, 0.214));
    MU_ASSERT("Rigidity force wrong in y", MU_ISCLOSE(dF.y, -0.971));
    MU_ASSERT("Rigidity force wrong in z", MU_ISCLOSE(dF.z, -1.802));
    MU_ASSERT("Rigidity force wrong in theta", MU_ISCLOSE(dF.theta, -0.339));
    MU_ASSERT("Rigidity force wrong in phi", MU_ISCLOSE(dF.phi, 0.453));

    return NULL;
}


__device__ Po_cell rigid_cubic_force(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > 1) return dF;

    auto F = 2 * (0.6 - dist) * (1 - dist) + powf(1 - dist, 2);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += rigidity_force(Xi, r, dist) * 0.2;
    return dF;
}

const char* test_line_of_four()
{
    Solution<Po_cell, 4, Tile_solver> bolls;

    for (auto i = 0; i < 4; i++) {
        bolls.h_X[i].x = 0.733333 * cosf((i - 0.5) * M_PI / 3);
        bolls.h_X[i].y = 0.733333 * sinf((i - 0.5) * M_PI / 3);
        bolls.h_X[i].z = 0;
        bolls.h_X[i].theta = M_PI / 2;
        bolls.h_X[i].phi = (i - 0.5) * M_PI / 3;
    }
    bolls.copy_to_device();
    auto com_i = center_of_mass(bolls);
    for (auto i = 0; i < 500; i++) {
        bolls.take_step<rigid_cubic_force>(0.5);
    }

    bolls.copy_to_host();
    for (auto i = 1; i < 4; i++) {
        auto prod = pol_scalar_product(bolls.h_X[0], bolls.h_X[i]);
        MU_ASSERT("Epithelial polarity not aligned", MU_ISCLOSE(prod, 1));
    }

    float3 r_01{bolls.h_X[1].x - bolls.h_X[0].x,
        bolls.h_X[1].y - bolls.h_X[0].y, bolls.h_X[1].z - bolls.h_X[0].z};
    float3 r_12{bolls.h_X[2].x - bolls.h_X[1].x,
        bolls.h_X[2].y - bolls.h_X[1].y, bolls.h_X[2].z - bolls.h_X[1].z};
    float3 r_23{bolls.h_X[3].x - bolls.h_X[2].x,
        bolls.h_X[3].y - bolls.h_X[2].y, bolls.h_X[3].z - bolls.h_X[2].z};
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_01.x, r_12.x));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_12.x, r_23.x));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_01.y, r_12.y));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_12.y, r_23.y));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_01.z, r_12.z));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_12.z, r_23.z));

    auto com_f = center_of_mass(bolls);
    MU_ASSERT("Momentum in line", MU_ISCLOSE(com_i.x, com_f.x));
    MU_ASSERT("Momentum in line", MU_ISCLOSE(com_i.y, com_f.y));
    MU_ASSERT("Momentum in line", MU_ISCLOSE(com_i.z, com_f.z));

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_pcp_force);
    MU_RUN_TEST(test_pcp);
    MU_RUN_TEST(test_rigidity_force);
    MU_RUN_TEST(test_line_of_four);
    return NULL;
}

MU_RUN_SUITE(all_tests);
