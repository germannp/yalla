#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/epithelium.cuh"
#include "minunit.cuh"


__device__ __managed__ Solution<pocell, 4, N2nSolver> X;


__device__ pocell epithelium(pocell Xi, pocell Xj, int i, int j) {
    pocell dF = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > 1) return dF;

    float F = 2*(0.6 - dist)*(1 - dist) + powf(1 - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    assert(dF.x == dF.x);  // For NaN f != f.

    dF += polarity_force(Xi, Xj)*0.2;
    return dF;
}

__device__ __managed__ nhoodint<pocell> potential = epithelium;

const char* test_line_of_four() {
    for (int i = 0; i < 4; i++) {
        X[i].x = 0.733333*cosf((i - 0.5)*M_PI/3);
        X[i].y = 0.733333*sinf((i - 0.5)*M_PI/3);
        X[i].z = 0;
        X[i].phi = (i - 0.5)*M_PI/3;
        X[i].theta = M_PI/2;
    }
    float3 com_i = center_of_mass(X);
    for (int i = 0; i < 500; i++) {
        X.step(0.5, potential);
    }

    for (int i = 1; i < 4; i++) {
        float prod = sinf(X[0].theta)*sinf(X[i].theta)*cosf(X[0].phi - X[i].phi)
            + cosf(X[0].theta)*cosf(X[i].theta);
        MU_ASSERT("Polarity not aligned", MU_ISCLOSE(prod, 1));
    }

    float3 r_01 = {X[1].x - X[0].x, X[1].y - X[0].y, X[1].z - X[0].z};
    float3 r_12 = {X[2].x - X[1].x, X[2].y - X[1].y, X[2].z - X[1].z};
    float3 r_23 = {X[3].x - X[2].x, X[3].y - X[2].y, X[3].z - X[2].z};
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_01.x, r_12.x));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_12.x, r_23.x));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_01.y, r_12.y));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_12.y, r_23.y));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_01.z, r_12.z));
    MU_ASSERT("Cells not on line", MU_ISCLOSE(r_12.z, r_23.z));

    float3 com_f = center_of_mass(X);
    MU_ASSERT("Momentum in line", MU_ISCLOSE(com_i.x, com_f.x));
    MU_ASSERT("Momentum in line", MU_ISCLOSE(com_i.y, com_f.y));
    MU_ASSERT("Momentum in line", MU_ISCLOSE(com_i.z, com_f.z));

    return NULL;
}


const char* all_tests() {
    MU_RUN_TEST(test_line_of_four);
    return NULL;
}

MU_RUN_SUITE(all_tests);
